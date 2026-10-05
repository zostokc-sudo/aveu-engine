"""Fusionne transcription + signaux comportementaux, puis fait interpréter
le tout par Claude pour produire un rapport structuré (BehaviorReport).

Deux principes de précision, appris en testant l'outil sur de vraies vidéos :

1. **Ligne de base personnelle.** Une valeur absolue de tension des sourcils
   ne veut rien dire seule — certaines personnes sont naturellement plus
   expressives que d'autres. On calcule donc, pour chaque signal continu, la
   médiane sur l'ensemble de l'enregistrement (= la ligne de base de CETTE
   personne), et on fournit à l'IA l'écart à cette ligne de base plutôt que la
   seule valeur brute. C'est le principe fondateur de la kinésique appliquée :
   calibrer sur l'individu, chercher les déviations.
2. **Désambiguïser le signal par la parole.** Parler active naturellement la
   mâchoire et les lèvres, sans lien avec une émotion. On calcule donc le
   taux de parole (speech_ratio) et la durée moyenne des pauses par fenêtre,
   pour que l'IA puisse discounter les signaux buccaux quand la personne
   parle activement, et leur donner plus de poids quand ils apparaissent en
   silence.
3. **La voix porte un signal indépendant du visage et du corps.** Hauteur
   (pitch), énergie et régularité de la voix (voir audio_analysis.py) sont
   calculées séparément et calibrées sur la même ligne de base personnelle
   que les signaux visuels — un canal de plus, avec la même règle de
   prudence : intensité mesurée, jamais cause affirmée.
"""

import json
import statistics
import sys
from pathlib import Path

from audio_analysis import AudioBucketSignals
from groq_llm import parse as llm_parse
from quality_check import precall_quality_warnings, quality_warnings
from schema import BehaviorReport, PreCallBrief
from transcribe import Segment
from visual_analysis import BucketSignals

# Version condensée à ~40% de la taille d'origine (voir git blame pour la
# version complète, plus pédagogique) : le tier gratuit de Groq plafonne à
# 8000 tokens/minute pour tout le compte, et la version longue dépassait ce
# plafond à elle seule avant même d'ajouter la timeline — testé en direct le
# 27/09/2026, erreur 413 tokens_per_minute avec la version complète sur un
# appel de 12 fenêtres seulement. Garde les RÈGLES opérantes, coupe la prose
# pédagogique/citations qui n'est pas nécessaire au modèle pour les appliquer.
SYSTEM_PROMPT = """Tu es un expert en analyse comportementale (kinésique, FACS, \
psychologie de la négociation B2B). La personne filmée est l'INTERLOCUTEUR \
EXTERNE (prospect/client), jamais un salarié du client (l'analyse d'émotions \
de salariés est interdite par l'AI Act européen, art. 5).

N'INVENTE JAMAIS DE SIGNAL hors de la liste fournie (visage, regard, \
sourcils, mâchoire, mains/gestes, posture, voix). Ne mentionne JAMAIS \
fréquence cardiaque, pouls, transpiration, dilatation des pupilles ou autre \
signal biométrique — rien de tout cela n'est mesuré.

Chaque fenêtre de la timeline (enregistrement en {context}) contient \
spoken_text (transcription SANS distinction des locuteurs), speech_ratio \
(audio) et visual_speech_ratio (lèvres de la personne filmée) — si \
speech_ratio est haut mais visual_speech_ratio bas, quelqu'un d'AUTRE parle \
(hors champ) : n'attribue pas le texte à la personne filmée, lis plutôt sa \
réaction non-verbale. avg_pause_ms anormalement long avant une réponse = \
charge cognitive probable. Les signaux *_vs_baseline (écart à la médiane \
personnelle sur tout l'enregistrement) priment TOUJOURS sur les valeurs \
absolues — une personne naturellement expressive a des valeurs absolues \
hautes en permanence sans que ce soit un signal.

Règles de lecture (aucun signal n'est probant seul) :
- smile + cheek_raise ENSEMBLE = sourire authentique (Duchenne) ; smile seul \
sans cheek_raise = social/politesse, pas une adhésion réelle.
- brow_tension seul est ambigu (concentration OU tension) ; combiné à \
jaw_tension haut et eye_wide bas = plutôt tension/désaccord.
- gaze_aversion : JAMAIS un indice de mensonge (DePaulo et al. 2003 : le \
regard est l'un des signaux les MOINS fiables pour la tromperie) — lis-le \
uniquement comme charge cognitive ou attention.
- Mains (fidget_score/fidget_peak, hand_to_face) : distingue les \
ILLUSTRATEURS (rythment la parole active, normaux) des ADAPTATEURS \
(auto-contact/gigotement SANS parole active — silence ou l'autre partie \
parle — plus significatifs, liés à un inconfort géré intérieurement). Un pic \
pendant que la personne ÉCOUTE (visual_speech_ratio bas) pèse plus qu'un pic \
pendant qu'elle parle et gesticule pour illustrer son propos. Plus fiable en \
pic qu'en moyenne.
- Conflit entre canaux (visage calme mais mains agitées, ou l'inverse) : \
pèse davantage le canal le moins consciemment contrôlé (mains/corps) — \
fuite non-verbale (Ekman & Friesen, 1969) — jamais comme certitude, une \
pondération.
- Voix : hausse nette de pitch_avg_hz/energy_avg par rapport à la baseline = \
emphase ou tension (contexte verbal pour trancher) ; chute brève d'énergie \
en milieu de phrase = hésitation ou mot choisi avec précaution.
- face_coverage bas = fiabilité réduite sur cette fenêtre, reste prudent.

RÈGLE CENTRALE — jamais de diagnostic de "stress" : une activation (gestes \
rapides, clignements fréquents, débit qui accélère, voix qui monte) a \
plusieurs causes aussi plausibles : stress, mais aussi excitation, envie de \
convaincre, forte implication (arousal ≠ valence, Russell 1980 — mesurer \
l'un ne dit rien sur l'autre). Pour trancher, regarde ce qui est DIT : \
argumenter/développer → engagement, pas gêne ; hésiter/éluder/changer de \
sujet → gêne probable. Décris le PATTERN observable ("semble vouloir appuyer \
son propos avec force"), jamais une étiquette clinique ("est stressé") — une \
personne qui se sent diagnostiquée se braque, une personne qui se sent \
comprise progresse.

Ta tâche : croiser verbal et non-verbal, calibré sur la personne elle-même, \
pour expliquer POURQUOI elle réagit ainsi à chaque moment clé — pas juste \
décrire. Un signal isolé n'est jamais une preuve ; si un signal ambigu n'a \
pas de confirmation croisée, baisse la confidence plutôt que d'affirmer.

L'OBJECTIF FINAL N'EST PAS DE DÉCRIRE, C'EST DE FAIRE AVANCER LE CLOSING. \
closing_verdict et coaching_recommendations doivent relier CHAQUE phrase à \
UNE ACTION précise pour le prochain échange, jamais rester au niveau du \
constat. Si buying_signal est haut mais deal_risk aussi, identifie CE QUI \
bloque précisément (key_moments à confidence basse, activation qui monte \
sans argumentation active) plutôt que de conclure vaguement "lever les \
freins".

VARIÉTÉ OBLIGATOIRE : si deux key_moments se lisent comme des variantes l'un \
de l'autre, SUPPRIME le moins informatif plutôt que de reformuler pour les \
faire paraître différents — 3 moments réellement distincts valent mieux que \
8 dont la moitié répète la même observation. Dis-le dans executive_summary \
si l'appel est globalement stable."""

# Signaux continus sur lesquels calculer une ligne de base personnelle
# (médiane sur tout l'enregistrement) plutôt que de ne fournir que la valeur
# absolue — voir le principe n°1 en tête de fichier.
_BASELINE_SIGNALS = [
    "smile", "cheek_raise", "mouth_frown", "brow_tension", "inner_brow_raise",
    "jaw_tension", "eye_wide", "gaze_aversion", "blink_rate_per_min", "fidget_score",
    "pitch_avg_hz", "energy_avg",
]


def _speech_ratio(segments: list[Segment], b_start: float, b_end: float) -> float:
    bucket_dur = b_end - b_start
    if bucket_dur <= 0:
        return 0.0
    covered = 0.0
    for s in segments:
        spans = [(w.start, w.end) for w in s.words] or [(s.start, s.end)]
        for w_start, w_end in spans:
            overlap = min(w_end, b_end) - max(w_start, b_start)
            if overlap > 0:
                covered += overlap
    return round(min(covered / bucket_dur, 1.0), 2)


def _avg_pause_ms(segments: list[Segment], b_start: float, b_end: float) -> float:
    words = sorted(
        (w for s in segments for w in s.words),
        key=lambda w: w.start,
    )
    pauses = []
    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        mid = (prev.end + nxt.start) / 2
        if gap > 0 and b_start <= mid < b_end:
            pauses.append(gap * 1000)
    return round(sum(pauses) / len(pauses), 0) if pauses else 0.0


def _bucket_signal_dict(b: BucketSignals) -> dict:
    return {
        "smile": round(b.smile_avg, 2),
        "smile_peak": round(b.smile_peak, 2),
        "cheek_raise": round(b.cheek_raise_avg, 2),
        "mouth_frown": round(b.mouth_frown_avg, 2),
        "brow_tension": round(b.brow_tension_avg, 2),
        "brow_tension_peak": round(b.brow_tension_peak, 2),
        "brow_tension_std": round(b.brow_tension_std, 3),
        "inner_brow_raise": round(b.inner_brow_raise_avg, 2),
        "jaw_tension": round(b.jaw_tension_avg, 2),
        "jaw_tension_peak": round(b.jaw_tension_peak, 2),
        "eye_wide": round(b.eye_wide_avg, 2),
        "blink_rate_per_min": round(b.blink_rate_per_min, 1),
        "gaze_aversion": round(b.gaze_aversion_avg, 2),
        "gaze_aversion_std": round(b.gaze_aversion_std, 3),
        "head_movement_deg": round(b.head_movement, 1),
        "hand_to_face_ratio": round(b.hand_to_face_ratio, 2),
        "torso_lean_deg": round(b.torso_lean_avg, 1),
        "fidget_score": round(b.fidget_score, 4),
        "fidget_peak": round(b.fidget_peak, 4),
        "visual_speech_ratio": round(b.visual_speech_ratio, 2),
    }


def _audio_signal_dict(a: AudioBucketSignals | None) -> dict:
    if a is None:
        return {"pitch_avg_hz": 0.0, "pitch_std_hz": 0.0, "energy_avg": 0.0,
                "energy_peak": 0.0, "energy_std": 0.0, "voiced_ratio": 0.0}
    return {
        "pitch_avg_hz": round(a.pitch_avg_hz, 1),
        "pitch_std_hz": round(a.pitch_std_hz, 1),
        "energy_avg": round(a.energy_avg, 4),
        "energy_peak": round(a.energy_peak, 4),
        "energy_std": round(a.energy_std, 4),
        "voiced_ratio": round(a.voiced_ratio, 2),
    }


def merge_timeline(
    segments: list[Segment],
    buckets: list[BucketSignals],
    audio_buckets: list[AudioBucketSignals] | None = None,
) -> list[dict]:
    usable = [b for b in buckets if b.face_coverage >= 0.3]
    if not usable:
        return []

    audio_by_start = {a.start: a for a in audio_buckets} if audio_buckets else {}

    signal_dicts = []
    for b in usable:
        d = _bucket_signal_dict(b)
        d.update(_audio_signal_dict(audio_by_start.get(b.start)))
        signal_dicts.append(d)

    baseline = {
        key: statistics.median(d[key] for d in signal_dicts)
        for key in _BASELINE_SIGNALS
    }

    timeline = []
    for b, signals in zip(usable, signal_dicts):
        spoken = " ".join(
            s.text for s in segments if s.start < b.end and s.end > b.start
        ).strip()
        for key in _BASELINE_SIGNALS:
            signals[f"{key}_vs_baseline"] = round(signals[key] - baseline[key], 3)

        timeline.append({
            "start_sec": round(b.start, 1),
            "end_sec": round(b.end, 1),
            "spoken_text": spoken,
            "speech_ratio": _speech_ratio(segments, b.start, b.end),
            "avg_pause_ms": _avg_pause_ms(segments, b.start, b.end),
            "signals": signals,
        })
    return timeline


# Un appel d'1h = ~720 fenêtres de 5s. Les envoyer toutes au LLM coûte cher
# et ralentit sans gagner en qualité : l'écrasante majorité des fenêtres d'un
# appel réel sont plates. Au-delà de ce seuil, on ne garde que les fenêtres
# les plus significatives — voir select_salient_windows.
_MAX_WINDOWS_FOR_LLM = 90


# La plupart des signaux vivent sur une échelle 0-1 ; les signaux vocaux ont
# des unités très différentes (Hz, amplitude RMS) qui écraseraient les autres
# dans une simple somme d'écarts. On normalise par un ordre de grandeur
# approximatif avant de sommer, pour que la voix pèse autant que le visage,
# pas plus.
_SALIENCE_SCALE = {"pitch_avg_hz": 50.0, "energy_avg": 0.02}


def _salience(window: dict) -> float:
    signals = window["signals"]
    deviation = sum(
        abs(signals.get(f"{key}_vs_baseline", 0.0)) / _SALIENCE_SCALE.get(key, 1.0)
        for key in _BASELINE_SIGNALS
    )
    peak_bonus = signals.get("fidget_peak", 0.0) + signals.get("brow_tension_peak", 0.0) + signals.get("jaw_tension_peak", 0.0)
    speech_bonus = 0.3 if window["spoken_text"] else 0.0
    return deviation + peak_bonus + speech_bonus


def select_salient_windows(timeline: list[dict], max_windows: int = _MAX_WINDOWS_FOR_LLM) -> list[dict]:
    """Réduit une timeline longue aux fenêtres les plus significatives (celles
    qui dévient le plus de la ligne de base personnelle, ou portent un pic),
    remises dans l'ordre chronologique. Ne change rien pour un appel court —
    ne s'active qu'au-delà de max_windows, donc le coût/temps reste borné
    quelle que soit la durée de l'enregistrement (1h ou plus)."""
    if len(timeline) <= max_windows:
        return timeline
    scored = sorted(timeline, key=_salience, reverse=True)[:max_windows]
    return sorted(scored, key=lambda w: w["start_sec"])


# Signaux que le SYSTEM_PROMPT condensé enseigne réellement à interpréter —
# voir le commentaire au-dessus de SYSTEM_PROMPT sur le plafond Groq à 8000
# tokens/minute. Envoyer un signal que le prompt ne couvre plus (ex:
# pitch_std_hz, torso_lean_deg) ne fait que consommer du budget token pour
# une donnée que le modèle n'a plus l'instruction d'exploiter. Les valeurs
# ABSOLUES des signaux à ligne de base (_BASELINE_SIGNALS) sont aussi omises
# ici : le prompt dit explicitement de prioriser l'écart à la baseline sur la
# valeur absolue, donc l'absolue est redondante pour ces signaux précis.
_PROMPT_SIGNAL_KEYS = [
    "smile_vs_baseline", "cheek_raise_vs_baseline",
    "brow_tension_vs_baseline", "brow_tension_peak",
    "jaw_tension_vs_baseline", "jaw_tension_peak",
    "eye_wide_vs_baseline",
    "gaze_aversion_vs_baseline", "gaze_aversion_std",
    "fidget_score_vs_baseline", "fidget_peak", "hand_to_face_ratio",
    "visual_speech_ratio",
    "pitch_avg_hz_vs_baseline", "energy_avg_vs_baseline", "energy_peak",
]


def _r(v):
    return round(v, 2) if isinstance(v, float) else v


def _compact_for_prompt(timeline: list[dict]) -> list[dict]:
    # Valeurs arrondies, signaux nuls omis, texte borné : chaque fenêtre pèse ~3x moins de tokens.
    return [
        {
            "start_sec": _r(w["start_sec"]), "end_sec": _r(w["end_sec"]),
            "spoken_text": w["spoken_text"][:220], "speech_ratio": _r(w["speech_ratio"]),
            "avg_pause_ms": _r(w["avg_pause_ms"]),
            "signals": {k: _r(w["signals"][k]) for k in _PROMPT_SIGNAL_KEYS if w["signals"].get(k)},
        }
        for w in timeline
    ]


# Plafond Groq gratuit : 8000 tokens/minute (entrée + sortie demandée). Budget en caractères
# (~3 caractères/token en français + JSON) pour le prompt entier, avec une sortie de 3200 tokens.
_GROQ_MAX_OUTPUT = 3200
_GROQ_PROMPT_CHARS = 13000


def _fit_timeline_to_budget(timeline: list[dict], system: str, intro: str) -> list[dict]:
    n = min(len(timeline), _MAX_WINDOWS_FOR_LLM)
    while True:
        chosen = select_salient_windows(timeline, n)
        size = len(system) + len(intro) + len(json.dumps(_compact_for_prompt(chosen), ensure_ascii=False))
        if size <= _GROQ_PROMPT_CHARS or n <= 8:
            return chosen
        n = max(8, int(n * 0.85))


def generate_report(
    segments: list[Segment],
    buckets: list[BucketSignals],
    context: str = "entretien de vente B2B",
    audio_buckets: list[AudioBucketSignals] | None = None,
) -> BehaviorReport:
    timeline = merge_timeline(segments, buckets, audio_buckets)
    if not timeline:
        raise ValueError("Aucune fenêtre exploitable (visage jamais détecté de façon fiable).")
    system = SYSTEM_PROMPT.format(context=context)
    intro = "Voici la timeline de l'enregistrement, en JSON. Produis le rapport d'analyse comportementale.\n\n"
    timeline = _fit_timeline_to_budget(timeline, system, intro)

    report = llm_parse(
        system=system,
        user=intro + json.dumps(_compact_for_prompt(timeline), ensure_ascii=False),
        output_format=BehaviorReport,
        max_tokens=_GROQ_MAX_OUTPUT,
    )
    for warning in quality_warnings(report, timeline):
        print(f"QUALITY WARNING: {warning}", file=sys.stderr)
    return report


def save_report_json(report: BehaviorReport, out_path: Path) -> None:
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


PRECALL_SYSTEM_PROMPT = """Tu prépares un commercial pour son PROCHAIN appel \
avec un interlocuteur externe (prospect/client) déjà rencontré. On te donne \
le rapport d'analyse comportementale du dernier échange avec cette même \
personne, en JSON (résumé, scores, moments clés horodatés, recommandations).

Ta tâche n'est pas de résumer ce rapport — c'est de préparer activement le \
prochain appel : quelles objections anticiper, quoi préparer concrètement \
avant d'y aller, comment ouvrir l'appel, et quoi surveiller en priorité dès \
le début. Base-toi strictement sur les signaux et moments du rapport fourni \
— n'invente pas de contexte qui n'y est pas. Reste concret et actionnable : \
"préparer un chiffrage précis du delta de prix évoqué" plutôt que "être \
transparent sur le prix". Même règle que pour l'analyse elle-même : jamais \
d'évaluation de l'employé qui a mené l'appel, uniquement une préparation \
pour mieux lire et répondre à l'interlocuteur externe.

Pour verification_questions : repère spécifiquement les key_moments à \
confidence "low" ou "medium" dans le rapport — ce sont des lectures où le \
signal non-verbal était ambigu sans confirmation verbale au moment même. Un \
signal ambigu doit générer une question à poser à l'oral au prochain appel, \
jamais une conclusion qu'on se contente de supposer vraie deux fois de \
suite. S'il n'y a aucun moment ambigu dans le rapport (tout est confidence \
"high"), retourne une liste vide plutôt que d'inventer une question générique.

Pour opening_move : si le rapport fait apparaître un point de friction \
identifiable, utilise la technique du LABELING (Chris Voss, "Never Split the \
Difference", 2016) — nommer explicitement ce qui semblait compter pour \
l'interlocuteur plutôt que de l'éviter ("On dirait que c'était surtout une \
question de validation en interne, plus que de prix"). Nommer ce qui \
préoccupe l'autre, sans l'accuser ni le lui reprocher, désarme la défense et \
invite à confirmer ou corriger — plus efficace qu'une ouverture neutre qui \
ignore ce qui s'est passé."""


def generate_precall_brief(previous_report: BehaviorReport, context: str = "prochain appel de vente") -> PreCallBrief:
    brief = llm_parse(
        system=PRECALL_SYSTEM_PROMPT,
        user=(
            f"Contexte du prochain appel : {context}\n\n"
            "Voici le rapport du dernier échange avec cet interlocuteur, en JSON :\n\n"
            + previous_report.model_dump_json()
        ),
        output_format=PreCallBrief,
        max_tokens=4000,
    )
    # Écrasés après coup plutôt que laissés au LLM : ce sont des chiffres déjà
    # connus avec certitude (copiés du rapport fourni), pas une estimation à
    # regénérer — voir schema.PreCallBrief.previous_deal_risk.
    brief.previous_deal_risk = previous_report.overall_scores.deal_risk
    brief.previous_buying_signal = previous_report.overall_scores.buying_signal
    for warning in precall_quality_warnings(brief):
        print(f"QUALITY WARNING: {warning}", file=sys.stderr)
    return brief


def compute_behavior_profile(buckets: list[BucketSignals]) -> dict[str, float]:
    """Axes déterministes (sans LLM) pour le radar visuel du rapport — décrivent
    des comportements observables (où regarde-t-on, sourit-on vraiment, les
    gestes sont-ils stables), jamais une cause psychologique supposée.

    gesture_stability suivait à l'origine un plafond absolu de vitesse des
    mains (fidget_score / 0.3, calibré à l'oeil sur un premier enregistrement
    de test très calme). Sur un vrai appel où la personne gesticule
    naturellement en parlant (vitesse moyenne observée jusqu'à 4-5x ce
    plafond), ce plafond saturait à 100% d'instabilité sur la totalité de
    l'appel — un score plat, sans aucun pouvoir discriminant, quelle que soit
    la variation réelle d'un moment à l'autre. Corrigé pour suivre le même
    principe de ligne de base personnelle que le reste du pipeline (voir
    l'en-tête du fichier) : on mesure la variabilité RELATIVE des gestes de
    cette personne au fil de CET appel (coefficient de variation — écart-type
    rapporté à la moyenne), pas leur vitesse absolue comparée à un seuil
    universel qui ne peut pas s'adapter à un style gestuel naturellement plus
    ample. Une personne qui gesticule beaucoup mais de façon constante reste
    donc "stable" ; un pic ponctuel qui tranche avec le reste de l'appel fait
    baisser le score, que l'amplitude absolue soit haute ou basse."""
    usable = [b for b in buckets if b.face_coverage >= 0.3]
    if not usable:
        return {"gaze_stability": 0.0, "genuine_smile": 0.0, "gesture_stability": 0.0}

    gaze_stability = 100 * (1 - statistics.mean(b.gaze_aversion_avg for b in usable))
    genuine_smile = 100 * statistics.mean((b.smile_avg + b.cheek_raise_avg) / 2 for b in usable)

    fidget_values = [b.fidget_score for b in usable]
    fidget_mean = statistics.mean(fidget_values)
    coefficient_of_variation = statistics.pstdev(fidget_values) / fidget_mean if fidget_mean > 1e-9 else 0.0
    gesture_stability = 100 * (1 - coefficient_of_variation)

    return {
        "gaze_stability": round(max(0.0, min(gaze_stability, 100.0)), 1),
        "genuine_smile": round(max(0.0, min(genuine_smile, 100.0)), 1),
        "gesture_stability": round(max(0.0, min(gesture_stability, 100.0)), 1),
    }
