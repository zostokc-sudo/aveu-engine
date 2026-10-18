"""Débrief du CLOSER à partir du TEXTE de l'échange (aucune émotion inférée, aucun visage nécessaire).

Pourquoi ce module existe : l'objectif d'Aveu est d'évaluer et d'améliorer le travail d'un closer. Ce qu'il dit
et fait pendant l'appel (ouverture, découverte, objections, proposition, conclusion) se lit dans la transcription ;
cela fonctionne même si la vidéo n'a aucun visage exploitable, et ne touche pas à l'interdiction de l'AI Act
(art. 5) puisqu'aucune émotion n'est lue sur personne.

Méthode en deux temps, calée sur le plafond gratuit de Groq (8000 tokens/minute) :
1. MAP    : la transcription est découpée en tranches ; pour chaque tranche le modèle relève des « notes »
            (questions de découverte, objections, tentatives de conclusion, occasions manquées…) avec citation.
2. REDUCE : les notes VÉRIFIÉES (citation retrouvée dans le texte) servent à rédiger le débrief final.
Toute citation qu'on ne retrouve pas dans la transcription est supprimée : jamais de phrase inventée."""

import json
import re
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from groq_llm import parse as llm_parse

DIMENSIONS = [
    ("opening", "Ouverture"), ("discovery", "Découverte"), ("listening", "Écoute"),
    ("objections", "Objections"), ("proposal", "Proposition"), ("closing", "Conclusion"),
]
_DIM_KEYS = Literal["opening", "discovery", "listening", "objections", "proposal", "closing"]
_KINDS = Literal[
    "opening", "discovery_question", "prospect_need", "buying_signal", "objection",
    "objection_response", "proposal", "closing_attempt", "next_step", "missed_opportunity", "monologue",
]


class Note(BaseModel):
    t_sec: int = Field(description="seconde où le passage commence (repère [mm:ss] de la transcription)")
    speaker: Literal["closer", "prospect", "inconnu"]
    kind: _KINDS
    quote: str = Field(description="citation VERBATIM de la transcription, 200 caractères maximum")
    comment: str = Field(description="en 15 mots maximum : pourquoi ce passage compte")


class ChunkNotes(BaseModel):
    notes: list[Note] = Field(description="10 notes maximum, les plus significatives de cette tranche")


class DimensionScore(BaseModel):
    key: _DIM_KEYS
    score: int = Field(description="0-100 : QUALITÉ D'EXÉCUTION de cette étape (pas une probabilité de signature)")
    verdict: str = Field(description="une phrase : ce qui a été bien ou mal fait, factuel")
    evidence_t_sec: int = Field(description="seconde de la citation (0 si aucune)")
    evidence_quote: str = Field(description="citation prise TELLE QUELLE dans les notes fournies, ou chaîne vide")
    improvement: str = Field(description="UNE action concrète, avec une formulation exacte que le closer peut dire la prochaine fois")


class Strength(BaseModel):
    t_sec: int
    quote: str = Field(description="citation prise telle quelle dans les notes")
    why: str = Field(description="pourquoi c'est un bon réflexe, en une phrase")


class MissedMoment(BaseModel):
    t_sec: int
    prospect_said: str = Field(description="ce que le prospect a dit (citation prise telle quelle dans les notes)")
    closer_did: str = Field(description="ce que le closer a fait ensuite, en une phrase")
    better_move: str = Field(description="la formulation exacte qu'il aurait pu employer")


class CloserDebrief(BaseModel):
    summary: str = Field(description="3 phrases maximum : comment le closer a mené cet appel, ton direct et bienveillant")
    scorecard: list[DimensionScore] = Field(description="exactement 6 entrées, une par dimension, dans l'ordre fourni")
    strengths: list[Strength] = Field(description="2 à 3 points forts réels")
    missed: list[MissedMoment] = Field(description="0 à 4 occasions manquées, les plus coûteuses d'abord")
    priorities: list[str] = Field(description="exactement 3 actions concrètes pour le PROCHAIN appel, par ordre d'impact")
    next_step_secured: bool = Field(description="une prochaine étape datée a-t-elle été fixée avant la fin ?")
    next_step_detail: str = Field(description="une phrase sur la fin d'appel")


_MAP_SYSTEM = """\
Tu relis une TRANCHE de la transcription d'un appel de vente, pour préparer le débrief du CLOSER (le vendeur).
Chaque ligne : [mm:ss] Locuteur: texte. Si le locuteur n'est pas indiqué, déduis qui est le closer (celui qui mène l'appel, \
pose les questions, présente l'offre) et qui est le prospect ; en cas de doute, mets "inconnu".

Relève au plus 10 notes, les plus utiles pour juger le TRAVAIL DU CLOSER : ouverture, questions de découverte, besoins ou \
douleurs exprimés par le prospect, signaux d'achat, objections et la façon d'y répondre, présentation de l'offre, \
tentatives de conclusion, prochaine étape fixée, occasions manquées (le prospect ouvre une porte et le closer ne la saisit \
pas), monologues trop longs.

RÈGLES : la citation doit être COPIÉE MOT POUR MOT de la tranche (jamais reformulée). Tu ne décris que des FAITS de langage : \
jamais d'émotion supposée, jamais de diagnostic psychologique, jamais de jugement sur la personne. Réponds en français."""

_REDUCE_SYSTEM = """\
Tu es un coach de closers. À partir de NOTES vérifiées extraites d'un appel de vente (citations réelles, avec leur seconde), \
rédige le débrief du CLOSER. Tu évalues son TRAVAIL (ce qu'il dit et fait), jamais sa personne, et tu n'inventes rien.

Dimensions à noter dans cet ordre : opening (ouverture), discovery (découverte), listening (écoute), objections, \
proposal (proposition), closing (conclusion). Le score mesure la qualité d'exécution de l'étape, pas la chance de signer. \
Si une étape est absente de l'appel, note-la bas et dis-le.

RÈGLES STRICTES :
- evidence_quote, quote et prospect_said : copiés TELS QUELS depuis les notes, sinon chaîne vide.
- improvement et better_move : une vraie phrase que le closer peut dire, naturelle, en français, adaptée à CET appel.
- Ton : direct, concret, bienveillant, tutoiement interdit (vouvoiement). Pas de jargon sans l'expliquer.
- Aucune émotion supposée, aucun diagnostic psychologique du prospect ou du closer.
- Exactement 6 entrées dans scorecard et exactement 3 priorités."""

_CHUNK_CHARS = 7000
_MAX_NOTES_REDUCE = 40
_SLEEP_BETWEEN_CALLS = 40  # secondes : garde le débit sous 8000 tokens/minute (plafond gratuit Groq)
_STOP = {"les", "des", "une", "que", "qui", "pas", "pour", "dans", "avec", "vous", "nous", "est", "sont", "the", "and", "you", "that", "this", "ça", "ce", "cette", "mais", "donc", "alors", "oui", "non"}


def _mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def transcript_text_lines(segments) -> list[tuple[float, str]]:
    labels = {"other": "Closer", "filmed": "Prospect"}
    out = []
    for s in segments:
        if not s.text.strip():
            continue
        who = labels.get(s.speaker)
        out.append((s.start, f"[{_mmss(s.start)}] {who + ': ' if who else ''}{s.text.strip()}"))
    return out


def chunk_lines(lines: list[tuple[float, str]], size: int = _CHUNK_CHARS) -> list[str]:
    chunks, cur, n = [], [], 0
    for _, line in lines:
        if n + len(line) > size and cur:
            chunks.append("\n".join(cur)); cur, n = [], 0
        cur.append(line); n += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wàâçéèêëîïôûùüÿœ'-]+", text.lower()) if len(w) > 2 and w not in _STOP}


def quote_is_grounded(quote: str, t_sec: float, segments, window: float = 150.0, threshold: float = 0.7) -> bool:
    """La citation (au moins 70 % de ses mots significatifs) se retrouve-t-elle dans le texte autour de t_sec ?"""
    q = _words(quote)
    if not q:
        return False
    near = _words(" ".join(s.text for s in segments if abs(s.start - t_sec) <= window))
    return len(q & near) / len(q) >= threshold


def _call(system: str, user: str, model_cls, max_tokens: int):
    """Appel Groq avec patience : le plafond gratuit se règle en attendant, pas en échouant."""
    last = None
    for attempt in range(5):
        try:
            return llm_parse(system=system, user=user, output_format=model_cls, max_tokens=max_tokens)
        except Exception as exc:
            last = exc
            msg = str(exc)
            if "rate" in msg.lower() or "429" in msg or "Request too large" in msg:
                time.sleep(35)
                continue
            if attempt == 0:
                continue  # le modèle rate parfois son JSON : une relance suffit
            break
    raise last


def collect_notes(segments, progress=print) -> list[Note]:
    lines = transcript_text_lines(segments)
    chunks = chunk_lines(lines)
    notes: list[Note] = []
    for i, chunk in enumerate(chunks):
        progress(f"Débrief : lecture de la tranche {i + 1}/{len(chunks)}…")
        try:
            result = _call(_MAP_SYSTEM, "Tranche de transcription :\n\n" + chunk, ChunkNotes, 1800)
            notes.extend(result.notes)
        except Exception as exc:
            print(f"Débrief : tranche {i + 1} ignorée ({exc!r})", file=sys.stderr)
        if i < len(chunks) - 1:
            time.sleep(_SLEEP_BETWEEN_CALLS)
    return [n for n in notes if quote_is_grounded(n.quote, n.t_sec, segments)]


def _select_notes(notes: list[Note], limit: int = _MAX_NOTES_REDUCE) -> list[Note]:
    priority = {"missed_opportunity": 0, "objection": 1, "objection_response": 1, "closing_attempt": 2, "next_step": 2,
                "buying_signal": 3, "proposal": 3, "prospect_need": 4, "discovery_question": 4, "opening": 5, "monologue": 5}
    ranked = sorted(notes, key=lambda n: (priority.get(n.kind, 6), n.t_sec))[:limit]
    return sorted(ranked, key=lambda n: n.t_sec)


def build_debrief(segments, roles_inferred: bool, progress=print) -> dict | None:
    """Renvoie le débrief (dict prêt pour le gabarit) ou None si le texte est trop pauvre."""
    if sum(len(s.text) for s in segments) < 600:
        return None
    notes = collect_notes(segments, progress)
    if len(notes) < 4:
        return None
    selected = _select_notes(notes)
    time.sleep(_SLEEP_BETWEEN_CALLS)
    progress("Débrief : rédaction…")
    payload = [{"t": _mmss(n.t_sec), "t_sec": n.t_sec, "qui": n.speaker, "type": n.kind, "citation": n.quote, "note": n.comment} for n in selected]
    for _ in range(4):  # si la requête dépasse le plafond, on garde moins de notes
        try:
            debrief = _call(_REDUCE_SYSTEM, "Notes de l'appel (JSON) :\n" + json.dumps(payload, ensure_ascii=False), CloserDebrief, 2600)
            break
        except Exception as exc:
            if "too large" in str(exc).lower() and len(payload) > 12:
                payload = payload[: int(len(payload) * 0.7)]
                continue
            raise
    return _sanitize(debrief, segments, roles_inferred)


def _sanitize(d: CloserDebrief, segments, roles_inferred: bool) -> dict:
    def ok(q: str, t: int) -> str:
        return q if q and quote_is_grounded(q, t, segments) else ""

    by_key = {s.key: s for s in d.scorecard}
    scorecard = []
    for key, label in DIMENSIONS:
        s = by_key.get(key)
        if not s:
            continue
        scorecard.append({
            "key": key, "label": label, "score": max(0, min(100, s.score)), "verdict": s.verdict,
            "quote": ok(s.evidence_quote, s.evidence_t_sec), "at": _mmss(s.evidence_t_sec) if s.evidence_t_sec else "",
            "improvement": s.improvement,
        })
    return {
        "summary": d.summary,
        "scorecard": scorecard,
        "strengths": [{"at": _mmss(x.t_sec), "quote": ok(x.quote, x.t_sec), "why": x.why} for x in d.strengths if ok(x.quote, x.t_sec)][:3],
        "missed": [{"at": _mmss(x.t_sec), "prospect_said": ok(x.prospect_said, x.t_sec), "closer_did": x.closer_did, "better_move": x.better_move}
                   for x in d.missed][:4],
        "priorities": d.priorities[:3],
        "next_step_secured": d.next_step_secured,
        "next_step_detail": d.next_step_detail,
        "roles_inferred": roles_inferred,
    }
