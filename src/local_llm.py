"""Génération de rapport via un petit modèle local (gratuit, tourne sur la
machine) — utilisé tant qu'aucune clé API Claude n'est configurée. Qualité
d'analyse inférieure à claude-opus-5, mais valide toute la chaîne mécanique.
"""

import gc
import json
import re

import mlx.core as mx
from mlx_lm import generate, load

from report_generator import merge_timeline, select_salient_windows  # noqa: F401 (ré-exportés pour webapp.py)
from schema import BehaviorReport, PreCallBrief

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Dans un processus dédié (voir llm_worker.py), rien d'autre ne consomme la
# mémoire Metal : on peut monter la limite bien plus haut que dans un
# processus partagé avec mediapipe (qui garde des buffers en cache après son
# passage, d'où le plafond bas historique et les OOM "Insufficient Memory").
mx.set_memory_limit(6 * 1024**3)

PROMPT_TEMPLATE = """Tu es un expert en analyse comportementale. N'invente JAMAIS \
un signal absent des données ci-dessous — en particulier jamais de fréquence \
cardiaque, pouls, transpiration, dilatation des pupilles ou autre signal \
biométrique : rien de tout cela n'est mesuré, même si c'est un signe de stress \
plausible en général. Limite-toi au visage, au regard, aux sourcils, à la \
mâchoire, aux mains/gestes, à la posture et à la voix. Voici la timeline \
d'un extrait d'appel : texte prononcé (attention, la transcription ne distingue \
pas les locuteurs). Pour savoir si c'est VRAIMENT la personne filmée qui parle, \
compare speech_ratio (audio, une voix est active) à visual_speech_ratio (ses \
lèvres bougent) : les deux hauts = c'est probablement elle ; speech_ratio haut \
mais visual_speech_ratio bas = quelqu'un d'autre parle (le commercial, hors \
champ) — dans ce cas ne lui attribue pas le texte, regarde plutôt sa réaction \
non-verbale. Décompte aussi les signaux de bouche/mâchoire quand \
visual_speech_ratio est élevé, ils viennent surtout de l'articulation. Les champs \
"*_vs_baseline" indiquent l'écart à la valeur médiane de CETTE personne sur \
tout l'enregistrement : base ton interprétation sur ces écarts, PAS sur les \
valeurs absolues — une personne naturellement expressive aura des valeurs \
absolues hautes en permanence sans que ce soit un signal. Les champs \
pitch_avg_hz/energy_avg/voiced_ratio viennent de la voix elle-même (hauteur, \
volume) — une hausse de hauteur ou de volume est un signal d'intensité, pas \
forcément négatif, à lire avec le même contexte que les autres signaux.

RÈGLE IMPORTANTE : ne jamais diagnostiquer du "stress". Une activation \
physiologique (gestes, clignements, débit) peut venir du stress, mais tout \
autant de l'excitation, de l'envie de convaincre ou d'une forte implication \
(arousal et valence sont deux dimensions indépendantes — modèle de Russell, \
1980). Regarde ce qui est dit pour trancher : si la personne argumente/\
développe, c'est probablement de l'engagement, pas de la gêne. Décris \
toujours le comportement observable ("semble vouloir appuyer son propos") \
plutôt qu'une étiquette clinique ("est stressé") — la personne doit pouvoir \
se reconnaître dans la description, pas se sentir diagnostiquée.

Distingue les gestes des mains selon Ekman & Friesen (1969) : ILLUSTRATEURS \
(gestes qui rythment la parole active, normaux, peu informatifs) vs \
ADAPTATEURS (auto-contact, gigotement SANS parler ou pendant que l'autre \
parle — pèse plus lourd, lié à l'auto-apaisement). Ne présente JAMAIS un \
regard fuyant (gaze_aversion) comme un signe de mensonge : la recherche \
(DePaulo et al., 2003) montre que c'est un indice peu fiable pour la \
tromperie — lis-le seulement comme charge cognitive ou attention.

{timeline}

Réponds UNIQUEMENT avec un objet JSON valide, rien d'autre, de cette forme exacte. \
IMPORTANT : "key_moments" contient AU MAXIMUM 8 entrées, mais PEUT EN CONTENIR \
MOINS — n'en mets pas une par fenêtre, choisis SEULEMENT les moments qui \
apportent une lecture vraiment différente des autres. Si plusieurs fenêtres \
montrent le même état (la personne reste stable), n'en garde qu'UNE SEULE, ne \
duplique pas la même observation avec des mots différents pour remplir la \
liste — 3 moments vraiment distincts valent mieux que 8 moments répétitifs. \
Garde chaque texte court (une phrase). Le JSON doit être complet et se \
terminer correctement, quitte à raccourcir les textes pour y arriver. \
"closing_verdict" et "coaching_recommendations" doivent faire AVANCER LE \
CLOSING, pas juste décrire : chaque phrase relie un signal précis observé à \
UNE ACTION concrète pour la suite, jamais une généralité ("soyez à \
l'écoute").
{{
  "executive_summary": "...",
  "overall_scores": {{"engagement": 0-100, "activation": 0-100, "openness": 0-100, "buying_signal": 0-100, "deal_risk": 0-100}},
  "key_moments": [{{"start_sec": 0, "end_sec": 5, "quote": "...", "behavioral_signal": "LIBELLÉ COURT, 5 mots max, ex: 'sourire + hausse du ton' (PAS une phrase complète, PAS les chiffres bruts, l'explication va dans interpretation)", "interpretation": "...", "confidence": "low|medium|high"}}],
  "closing_verdict": "UNE phrase : où en est ce deal et quelle action prioritaire fait avancer le closing maintenant",
  "coaching_recommendations": ["action concrète sourcée par un moment précis, pas une généralité"]
}}
"""

_model = None
_tokenizer = None


def _get_model(model_id: str = MODEL_ID):
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load(model_id)
    return _model, _tokenizer


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


# Modèle local beaucoup plus limité en contexte/vitesse que claude-opus-5 sur
# cette machine à 8 Go de RAM — on garde moins de fenêtres qu'en production
# pour rester rapide et éviter l'OOM Metal sur un appel long. Abaissé de 40 à
# 20 le 2026-09-15 : l'enrichissement des signaux (audio, visual_speech_ratio,
# baselines) a fait grossir chaque fenêtre — 38 clés/fenêtre contre ~19 au
# moment où 40 avait été validé, ce qui a fait replanter l'OOM sur un vrai
# appel long malgré la limite mémoire à 6 Go.
_MAX_WINDOWS_LOCAL = 20


class GenerationTruncated(RuntimeError):
    """Le modèle local a été coupé avant de finir son JSON (max_tokens trop bas)."""


def generate_report_local(timeline: list[dict], model_id: str = MODEL_ID) -> BehaviorReport:
    timeline = select_salient_windows(timeline, max_windows=_MAX_WINDOWS_LOCAL)
    gc.collect()
    mx.clear_cache()
    model, tokenizer = _get_model(model_id)
    prompt = PROMPT_TEMPLATE.format(timeline=json.dumps(timeline, ensure_ascii=False))
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=formatted, max_tokens=3000, verbose=False)
    mx.clear_cache()
    try:
        return BehaviorReport.model_validate_json(_extract_json(raw))
    except ValueError as e:
        raise GenerationTruncated(
            "Le modèle local a produit un JSON incomplet (probablement coupé "
            "avant la fin) — réessaie, ou utilise un extrait plus court."
        ) from e


PRECALL_PROMPT_TEMPLATE = """Tu prépares un commercial pour son PROCHAIN appel \
avec un interlocuteur déjà rencontré. Voici le rapport du dernier échange \
avec cette personne (JSON) : {report}

Contexte du prochain appel : {context}

Prépare le prochain appel à partir de ce rapport — pas un résumé, une \
préparation active : objections probables, actions concrètes à préparer \
avant d'y aller, comment ouvrir l'appel, quoi surveiller en priorité. \
Base-toi uniquement sur ce que contient le rapport, n'invente rien.

Pour verification_questions : repère les key_moments à confidence "low" ou \
"medium" (signal ambigu, jamais confirmé à l'oral) et transforme chacun en \
une question concrète à poser au prochain appel — un doute non-verbal doit \
devenir une question, jamais une conclusion qu'on suppose vraie sans vérifier. \
Si tous les moments sont confidence "high", liste vide.

Pour opening_move : si un point de friction ressort du rapport, nomme-le \
explicitement dès l'ouverture (technique du "labeling", Chris Voss) — ex. "On \
dirait que c'était surtout une question de validation interne" — plutôt \
qu'une ouverture neutre qui l'ignore. Ne jamais accuser, juste nommer.

Réponds UNIQUEMENT avec un objet JSON valide, rien d'autre, de cette forme exacte :
{{
  "context_recap": "2-3 phrases sur ce qui s'est passé au dernier échange",
  "likely_objections": ["...", "..."],
  "prep_actions": ["...", "..."],
  "opening_move": "...",
  "watch_for": ["...", "..."],
  "verification_questions": ["question pour lever un doute resté ambigu, ou liste vide"],
  "previous_deal_risk": 0,
  "previous_buying_signal": 0
}}
"""


def generate_precall_brief_local(previous_report: BehaviorReport, context: str = "prochain appel de vente", model_id: str = MODEL_ID) -> PreCallBrief:
    gc.collect()
    mx.clear_cache()
    model, tokenizer = _get_model(model_id)
    prompt = PRECALL_PROMPT_TEMPLATE.format(report=previous_report.model_dump_json(), context=context)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=formatted, max_tokens=1500, verbose=False)
    mx.clear_cache()
    try:
        brief = PreCallBrief.model_validate_json(_extract_json(raw))
    except ValueError as e:
        raise GenerationTruncated(
            "Le modèle local a produit un JSON incomplet pour la fiche de "
            "préparation — réessaie."
        ) from e
    # Écrasés après coup, voir schema.PreCallBrief.previous_deal_risk.
    brief.previous_deal_risk = previous_report.overall_scores.deal_risk
    brief.previous_buying_signal = previous_report.overall_scores.buying_signal
    return brief
