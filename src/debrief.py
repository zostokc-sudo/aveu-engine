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
    is_sales_conversation: bool = Field(description="vrai si cette tranche est un ÉCHANGE entre un vendeur et un prospect (besoin, offre, prix, objection…) ; faux pour une conférence, un cours, un monologue ou un échange sans enjeu de vente")
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
Chaque ligne : [mm:ss] Locuteur: texte. Les étiquettes Closer/Prospect viennent d'une détection automatique qui peut se tromper (ou s'inverser) : si le CONTENU montre l'inverse \
(celui qui pose les questions et présente l'offre est le closer), corrige-le dans le champ speaker. Si le locuteur n'est pas indiqué, déduis qui est le closer (celui qui mène l'appel, \
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

GRILLE DE MÉTHODE (appuie ton jugement dessus, sans jargon ni noms de méthodes dans le texte) :
- Ouverture : cadre posé (objectif, durée, déroulé), intro courte et professionnelle, le prospect parle vite.
- Découverte : questions ouvertes, creuser la situation, le problème, ses CONSÉQUENCES (« que se passe-t-il si rien ne change ? »), « pourquoi maintenant ? » ; idéalement la majorité de l'appel.
- Écoute : le prospect parle plus que le closer, reformulations, relances sur ses mots.
- Objections : les anticiper tôt, les écouter, les reconnaître, les explorer AVANT de répondre ; ne pas les esquiver ni parler d'argent à contrecœur.
- Proposition : relier l'offre au problème et au résultat voulu par le prospect (pas lister des fonctionnalités) ; annoncer le prix puis une question ouverte.
- Conclusion : récapituler avec ses mots, demander clairement la décision, fixer une prochaine étape DATÉE (idéalement paiement ou rendez-vous immédiat).
Ne recommande jamais la culpabilisation, l'humiliation, la fausse urgence ni la pression sur la famille du prospect.

RÈGLES STRICTES :
- evidence_quote, quote et prospect_said : copiés TELS QUELS depuis les notes, sinon chaîne vide.
- improvement et better_move : une vraie phrase que le closer peut dire, naturelle, en français, adaptée à CET appel.
- Ton : direct, concret, bienveillant, tutoiement interdit (vouvoiement). Pas de jargon sans l'expliquer.
- Aucune émotion supposée, aucun diagnostic psychologique du prospect ou du closer.
- Exactement 6 entrées dans scorecard et exactement 3 priorités.
- COHÉRENCE : le score doit refléter le verdict (un verdict qui pointe un défaut ⇒ score inférieur à 55 ; une étape bien menée ⇒ au moins 70).
- Pour juger l'OUVERTURE et la CONCLUSION, appuie-toi d'abord sur les extraits bruts du début et de la fin de l'appel fournis ; \
n'écris jamais « aucune ouverture » ou « aucune conclusion » si ces extraits en montrent une."""

_CHUNK_CHARS = 7000
_MAX_NOTES_REDUCE = 32
_EDGE_CHARS = 900  # extraits bruts du début et de la fin, pour juger l'ouverture et la conclusion sur le texte réel
_SLEEP_BETWEEN_CALLS = 40  # secondes : garde le débit sous 8000 tokens/minute (plafond gratuit Groq)
_STOP = {"les", "des", "une", "que", "qui", "pas", "pour", "dans", "avec", "vous", "nous", "est", "sont", "the", "and", "you", "that", "this", "ça", "ce", "cette", "mais", "donc", "alors", "oui", "non"}


def _mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def transcript_text_lines(segments) -> list[tuple[float, str]]:
    labels = {"other": "Closer", "filmed": "Prospect"}  # rôles : "filmed" = prospect, "other" = closer
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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wàâçéèêëîïôûùüÿœ'-]+", text.lower().replace("\u2019", "'"))


def _words(text: str) -> set[str]:
    return {w for w in _tokens(text) if len(w) > 2 and w not in _STOP}


def quote_is_grounded(quote: str, t_sec: float, segments, window: float = 150.0, threshold: float = 0.7, speaker: str | None = None) -> bool:
    """La citation se retrouve-t-elle dans le texte autour de t_sec ?
    Il faut (1) au moins 70 % de ses mots significatifs ET (2) au moins 40 % de ses enchaînements de deux mots consécutifs
    (une paraphrase qui réutilise des mots épars est écartée). `speaker` limite la recherche aux phrases d'un rôle."""
    q = _tokens(quote)
    if len(q) < 2:
        return False
    near_segs = [s for s in segments if abs(s.start - t_sec) <= window and (speaker is None or s.speaker == speaker)]
    near = _tokens(" ".join(s.text for s in near_segs))
    sig = {w for w in q if len(w) > 2 and w not in _STOP} or set(q)  # citation faite de mots courts : on compare tous les mots
    near_set = set(near)
    if len(sig & near_set) / len(sig) < threshold:
        return False
    big_q = list(zip(q, q[1:]))
    big_n = set(zip(near, near[1:]))
    return sum(b in big_n for b in big_q) / len(big_q) >= 0.4


_LABEL = re.compile(r"^\s*(?:\[\d{1,2}:\d{2}\]\s*)?(?:closer|prospect)\s*:\s*", re.I)


def clean_quote(quote: str) -> str:
    """Le modèle recopie parfois « [03:12] Prospect : » devant une citation : on enlève l'étiquette."""
    return _LABEL.sub("", quote).strip()


def locate_quote(quote: str, segments, speaker: str | None = None, threshold: float = 0.7) -> float | None:
    """Cherche la citation dans toute la transcription (le modèle se trompe souvent de seconde) et renvoie le début
    du passage où elle se trouve vraiment, ou None si elle n'existe pas. Mêmes exigences que quote_is_grounded."""
    q = _tokens(clean_quote(quote))
    if len(q) < 2:
        return None
    sig = {w for w in q if len(w) > 2 and w not in _STOP} or set(q)
    big_q = list(zip(q, q[1:]))
    cands = [s for s in segments if (speaker is None or s.speaker == speaker) and s.text.strip()]

    def score(text: str) -> float:
        near = _tokens(text)
        if len(sig & set(near)) / len(sig) < threshold:
            return 0.0
        big_n = set(zip(near, near[1:]))
        ratio = sum(b in big_n for b in big_q) / len(big_q)
        return ratio if ratio >= 0.4 else 0.0

    best, best_score = None, 0.0
    for seg in cands:  # d'abord un passage seul (le plus précis)
        r = score(seg.text)
        if r > best_score:
            best, best_score = seg.start, r
    if best is not None:
        return best
    for seg, nxt in zip(cands, cands[1:]):  # sinon la citation chevauche deux passages consécutifs
        r = score(seg.text + " " + nxt.text)
        if r > best_score:
            best, best_score = seg.start, r
    return best


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 or "rate_limit" in msg or "rate limit" in msg or "request too large" in msg


def _call(system: str, user: str, model_cls, max_tokens: int):
    """Appel Groq avec patience : le plafond gratuit se règle en attendant, pas en échouant."""
    last = None
    for attempt in range(5):
        try:
            return llm_parse(system=system, user=user, output_format=model_cls, max_tokens=max_tokens)
        except Exception as exc:
            last = exc
            if _is_rate_limit(exc) and attempt < 4:
                time.sleep(35)
                continue
            if attempt == 0 and not _is_rate_limit(exc):
                continue  # le modèle rate parfois son JSON : une relance suffit
            break
    raise last


class NotASalesCall(RuntimeError):
    """L'enregistrement n'est pas un appel de vente (conférence, cours, monologue…) : pas de débrief de closer."""


def collect_notes(segments, progress=print) -> list[Note]:
    votes: list[bool] = []
    lines = transcript_text_lines(segments)
    chunks = chunk_lines(lines)
    notes: list[Note] = []
    for i, chunk in enumerate(chunks):
        progress(f"Débrief : lecture de la tranche {i + 1}/{len(chunks)}…")
        try:
            result = _call(_MAP_SYSTEM, "Tranche de transcription :\n\n" + chunk, ChunkNotes, 1800)
            notes.extend(result.notes)
            votes.append(result.is_sales_conversation)
        except Exception as exc:
            print(f"Débrief : tranche {i + 1} ignorée ({exc!r})", file=sys.stderr)
        if i < len(chunks) - 1:
            time.sleep(_SLEEP_BETWEEN_CALLS)
    if votes and sum(votes) / len(votes) < 0.5:
        raise NotASalesCall("Cet enregistrement ne ressemble pas à un appel de vente (conférence, cours ou monologue).")
    kept = []
    for n in notes:
        n.quote = clean_quote(n.quote)
        at = locate_quote(n.quote, segments)
        if at is not None:
            n.t_sec = int(at)  # la seconde réelle remplace celle (souvent approximative) du modèle
            kept.append(n)
    return kept


def _select_notes(notes: list[Note], limit: int = _MAX_NOTES_REDUCE) -> list[Note]:
    priority = {"missed_opportunity": 0, "objection": 1, "objection_response": 1, "closing_attempt": 2, "next_step": 2,
                "buying_signal": 3, "proposal": 3, "prospect_need": 4, "discovery_question": 4, "opening": 5, "monologue": 5}
    ranked = sorted(notes, key=lambda n: (priority.get(n.kind, 6), n.t_sec))[:limit]
    return sorted(ranked, key=lambda n: n.t_sec)


def build_debrief(segments, roles_inferred: bool, progress=print, focus=()) -> dict | None:
    """Renvoie le débrief (dict prêt pour le gabarit) ou None si le texte est trop pauvre."""
    if sum(len(s.text) for s in segments) < 600:
        return None
    notes = collect_notes(segments, progress)
    if len(notes) < 4:
        return None
    selected = _select_notes(notes)
    time.sleep(_SLEEP_BETWEEN_CALLS)
    progress("Débrief : rédaction…")
    lines = [line for _, line in transcript_text_lines(segments)]
    joined = "\n".join(lines)
    edges = f"Début de l'appel (extrait brut) :\n{joined[:_EDGE_CHARS]}\n\nFin de l'appel (extrait brut) :\n{joined[-_EDGE_CHARS:]}\n\n"
    focus_keys = [k for k, _ in DIMENSIONS if k in set(focus)]
    focus_note = ""
    if focus_keys:
        names = ", ".join(label for k, label in DIMENSIONS if k in focus_keys)
        focus_note = (f"FOCUS DEMANDÉ par le closer : {names}. Pour ces étapes uniquement, 'improvement' contient DEUX formulations "
                      "différentes séparées par « — ou — », et 'verdict' est plus précis (2 phrases). Les autres étapes restent brèves.\n\n")
    def to_payload(sel):
        return [{"t": _mmss(n.t_sec), "t_sec": n.t_sec, "qui": n.speaker, "type": n.kind, "citation": n.quote, "note": n.comment} for n in sel]

    limit = len(selected)
    payload = to_payload(selected)
    out_tokens = 3000 if focus_keys else 2600
    debrief = None
    for _ in range(5):  # le modèle peut rater son JSON, dépasser le plafond gratuit ou oublier une étape : on ajuste puis on réessaie
        problem = None
        try:
            debrief = _call(_REDUCE_SYSTEM, focus_note + edges + "Notes de l'appel (JSON) :\n" + json.dumps(payload, ensure_ascii=False), CloserDebrief, out_tokens)
            if len({d.key for d in debrief.scorecard}) == len(DIMENSIONS):
                break
            problem = "scorecard incomplet"
            debrief = None
        except Exception as exc:
            msg = str(exc).lower()
            if not ("too large" in msg or "json_validate_failed" in msg or "max completion" in msg or "truncated" in msg):
                raise
            problem = "réponse invalide ou trop grosse"
        out_tokens = min(out_tokens + 600, 3600)
        if limit > 10 and problem != "scorecard incomplet":
            limit = int(limit * 0.75)
            payload = to_payload(_select_notes(notes, limit))  # on retire les notes les moins décisives, pas la fin de l'appel
        print(f"Débrief : {problem}, nouvel essai ({len(payload)} notes, {out_tokens} tokens de sortie max)")
        time.sleep(30)
    if debrief is None:
        raise RuntimeError("Le modèle n'a pas pu rédiger un débrief complet.")
    return _sanitize(debrief, segments, roles_inferred, focus_keys)


def _sanitize(d: CloserDebrief, segments, roles_inferred: bool, focus_keys=()) -> dict:
    def ok(q: str, speaker: str | None = None) -> tuple[str, float | None]:
        q = clean_quote(q) if q else q
        at = locate_quote(q, segments, speaker) if q else None
        return (q, at) if at is not None else ("", None)

    by_key = {s.key: s for s in d.scorecard}
    scorecard = []
    for key, label in DIMENSIONS:
        s = by_key.get(key)
        if not s:
            continue
        quote, at = ok(s.evidence_quote)
        scorecard.append({
            "key": key, "label": label, "score": max(0, min(100, s.score)), "verdict": s.verdict,
            "quote": quote, "at": _mmss(at) if at is not None else "",
            "improvement": s.improvement, "focus": key in focus_keys,
        })
    strengths = []
    for x in d.strengths:
        quote, at = ok(x.quote)
        if quote:
            strengths.append({"at": _mmss(at), "quote": quote, "why": x.why})
    missed = []
    for x in d.missed:
        said, at = ok(x.prospect_said)
        if said:  # sans citation vérifiée, pas d'occasion manquée affichée
            missed.append({"at": _mmss(at), "prospect_said": said, "closer_did": x.closer_did, "better_move": x.better_move})
    return {
        "summary": d.summary,
        "scorecard": scorecard,
        "strengths": strengths[:3],
        "missed": missed[:4],
        "priorities": d.priorities[:3],
        "next_step_secured": d.next_step_secured,
        "next_step_detail": d.next_step_detail,
        "roles_inferred": roles_inferred,
        "focus_labels": [label for k, label in DIMENSIONS if k in focus_keys],
    }
