"""Génère les coordonnées SVG d'un radar (araignée) à partir d'une liste
d'axes 0-100, avec un code couleur rouge/orange/vert par axe. Pas de
dépendance externe (pas de lib de chart, pas de CDN) — juste de la
trigonométrie, pour rester 100% autonome et gratuit dans le rapport HTML.

Chaque axe porte aussi une explication fixe (ce qu'il mesure, pourquoi c'est
pertinent pour closer) et un commentaire selon la zone obtenue (rouge/orange/
vert) — écrits une fois ici plutôt que générés par le LLM à chaque rapport :
le sens d'un axe ne change pas d'un appel à l'autre, seule la valeur change,
donc laisser le LLM le réexpliquer à chaque fois n'ajoute que du risque de
fabrication pour aucun gain (voir quality_check.py, très strict sur ce point
ailleurs dans le pipeline)."""

import math
from dataclasses import dataclass

RED = "#c0392b"
ORANGE = "#d68910"
GREEN = "#1e8449"
NEUTRAL = "#5b6472"


@dataclass
class RadarAxis:
    key: str  # identifiant stable (voir _AXIS_INFO) — indépendant du label affiché
    label: str
    value: float  # 0-100
    higher_is_better: bool = True
    neutral: bool = False  # True = pas de jugement bon/mauvais (ex: activation)


# Contenu fixe par axe : ce qu'il mesure concrètement + pourquoi il compte
# pour faire avancer un closing — jamais généré par le LLM (voir docstring).
_AXIS_INFO = {
    "engagement": {
        "what": "Niveau d'implication active dans l'échange : attention portée, réactivité, participation — pas une simple présence passive.",
        "why": "Un closing tenté sans engagement réel n'a pas de terrain : la personne n'est pas assez impliquée pour s'engager sur une suite concrète.",
        "zones": {
            "red": "Engagement faible sur l'ensemble de l'échange — le terrain n'est pas prêt pour une tentative de closing ; regagner l'attention avant de proposer une suite.",
            "orange": "Engagement présent mais pas total — une tentative de closing est possible mais risquée sans avoir d'abord testé l'intérêt sur un point précis.",
            "green": "Engagement fort tout au long de l'échange — le terrain est favorable pour une proposition concrète.",
        },
    },
    "openness": {
        "what": "Disposition à l'écoute et confiance projetée : posture ouverte, sourire, contact visuel stable.",
        "why": "L'ouverture est la porte d'entrée du closing : sans elle, toute proposition ferme arrive trop tôt et risque d'être perçue comme une pression.",
        "zones": {
            "red": "Posture fermée dominante — introduire une offre ferme maintenant risque de braquer davantage ; travailler d'abord la relation.",
            "orange": "Ouverture partielle — le terrain accepte une proposition mesurée, pas encore un closing direct.",
            "green": "Ouverture nette — le moment est favorable pour introduire une étape concrète vers la signature.",
        },
    },
    "buying_signal": {
        "what": "Force des signaux verbaux et non-verbaux d'accord, d'adhésion ou d'intérêt réel à avancer.",
        "why": "C'est LE signal direct de closing : un score élevé est le feu vert pour proposer l'étape suivante précise (date, chiffrage, contrat) — attendre plus longtemps coûte de l'élan.",
        "zones": {
            "red": "Signal d'achat faible — proposer une signature maintenant serait prématuré ; identifier d'abord ce qui retient l'adhésion.",
            "orange": "Signal d'achat présent mais mêlé de réserve — tester l'engagement sur un point précis avant de proposer l'étape finale.",
            "green": "Signal d'achat fort — c'est le moment de proposer explicitement l'étape suivante, ne pas laisser retomber cet élan.",
        },
    },
    "deal_risk": {
        "what": "Probabilité que l'affaire échoue, traîne ou se rouvre plus tard, déduite des tensions et hésitations observées.",
        "why": "Un risque élevé impose de traiter l'objection sous-jacente avant toute tentative de closing — pousser malgré un risque non traité braque l'interlocuteur plutôt que de le convaincre.",
        "zones": {
            "red": "Risque élevé détecté — un point de friction n'est probablement pas résolu ; le nommer explicitement avant de reparler d'avancer est plus efficace que de l'ignorer.",
            "orange": "Risque modéré — surveiller si la tension observée se confirme ou se dissipe au prochain échange avant de pousser vers la signature.",
            "green": "Risque faible — rien n'indique de frein caché, le chemin vers la signature semble dégagé sur ce plan.",
        },
    },
    "gaze_stability": {
        "what": "Régularité du contact visuel au fil de l'échange — PAS un détecteur de mensonge : un regard qui se détourne se lit comme une charge cognitive, jamais comme une preuve de sincérité douteuse (DePaulo et al., 2003).",
        "why": "Un regard stable pendant les moments clés confirme une présence réelle sur le sujet ; une instabilité localisée signale un moment à repasser en douceur, pas un signal à interpréter seul.",
        "zones": {
            "red": "Regard instable sur une bonne partie de l'échange — probablement de la charge cognitive à des moments précis, à recroiser avec ce qui se disait plutôt qu'à interpréter seul.",
            "orange": "Regard globalement stable avec quelques instabilités ponctuelles — vérifier si elles coïncident avec un sujet sensible précis.",
            "green": "Regard stable tout au long de l'échange — pas de signal d'alerte à recroiser sur ce plan.",
        },
    },
    "genuine_smile": {
        "what": "Sourire « de Duchenne » (muscles autour des yeux activés) plutôt qu'un sourire social/de politesse — distinction établie par Ekman, Davidson & Friesen (1990).",
        "why": "Un sourire authentique au moment où on parle prix ou engagement est un signal d'adhésion réelle ; un sourire de politesse au même moment ne doit pas être confondu avec un accord.",
        "zones": {
            "red": "Peu de sourire authentique détecté — la courtoisie observée ne doit pas être lue comme de l'adhésion.",
            "orange": "Sourire authentique présent par moments — à recroiser avec les moments clés pour voir à quels sujets il est associé.",
            "green": "Sourire authentique fréquent — bon indicateur d'adhésion réelle, pas seulement de politesse sociale.",
        },
    },
    "gesture_stability": {
        "what": "Variabilité RELATIVE des gestes par rapport à la ligne de base personnelle de CET échange — pas une vitesse absolue : une gestuelle ample mais constante reste « stable » au sens de ce score.",
        "why": "Une instabilité gestuelle localisée à un moment précis signale un pic d'activation à corréler avec ce qui se disait à cet instant — pas une conclusion en soi sur l'ensemble de l'échange.",
        "zones": {
            "red": "Gestuelle marquée par un ou plusieurs pics d'instabilité nets — à recroiser avec les moments clés correspondants plutôt qu'à généraliser à tout l'échange.",
            "orange": "Quelques variations gestuelles notables — probablement liées à des moments précis, sans motif dominant clair.",
            "green": "Gestuelle stable tout au long de l'échange, quelle que soit son amplitude — pas de pic d'activation gestuelle isolé à signaler.",
        },
    },
}


_LLM_HOW = (
    "Score de 0 à 100 attribué par l'IA après lecture croisée de la transcription et des signaux mesurés "
    "(visage, corps, voix), recalés sur la ligne de base propre à cette personne. C'est une estimation "
    "argumentée, pas une mesure physique : les moments clés du rapport montrent sur quoi elle s'appuie."
)

# Court (survol), calcul (clic) et pistes de travail (clic) — textes fixes, jamais générés par le LLM.
_AXIS_EXTRA = {
    "engagement": {
        "how_short": "Estimé par l'IA en lisant ce qui se dit et comment la personne réagit.",
        "work_short": ['Posez une question ouverte toutes les 2-3 min', "Reformulez ce qu'elle dit avant de proposer la suite", 'Repérez où son attention a chuté'],
        "short": "À quel point la personne est impliquée dans l'échange, pas juste présente.",
        "how": _LLM_HOW,
        "work": ["Poser une question ouverte tous les 2-3 minutes pour relancer la participation.", "Repérer dans les moments clés quand l'attention a chuté et ce qui se disait à cet instant.", "Reformuler ce qu'elle dit avant de proposer la suite : ça montre qu'on l'écoute et ça la réengage."],
    },
    "openness": {
        "how_short": "Estimé par l'IA : posture, regard, sourire et ton, croisés avec les mots.",
        "work_short": ["Commencez par le relationnel avant l'offre", 'Posture fermée ? Posez une question sur son contexte', "Notez les sujets où elle s'ouvre"],
        "short": "Est-ce qu'elle est réceptive : posture ouverte, regard, sourire.",
        "how": _LLM_HOW,
        "work": ["Commencer par le relationnel avant toute offre ferme.", "Si la posture est fermée, ralentir et poser une question sur son contexte plutôt que pousser.", "Noter à quels sujets elle s'ouvre ou se ferme : ce sont ses vrais centres d'intérêt."],
    },
    "buying_signal": {
        "how_short": "Estimé par l'IA : accords, intérêt, phrases où la personne se projette.",
        "work_short": ['Score haut : proposez une étape précise, tout de suite', 'Score bas : demandez ce qui manque pour avancer', 'Cherchez les « quand on… » dans les moments clés'],
        "short": "Les signes qu'elle est prête à avancer : accord, intérêt, projection.",
        "how": _LLM_HOW,
        "work": ["Quand le score est haut, proposer tout de suite une étape précise (date, devis, contrat).", "Quand il est bas, demander ce qui manque pour avancer plutôt que de relancer l'offre.", "Chercher dans les moments clés les phrases où elle se projette (« quand on… », « chez nous… »)."],
    },
    "deal_risk": {
        "how_short": "Estimé par l'IA : tensions et hésitations repérées. Ici, plus c'est bas, mieux c'est.",
        "work_short": ['Nommez le frein à voix haute avant de parler signature', "Traitez d'abord l'objection restée sans réponse", 'Préparez le prochain appel avec la fiche dédiée'],
        "short": "Le risque que l'affaire échoue, traîne ou se rouvre. Plus c'est bas, mieux c'est.",
        "how": _LLM_HOW + " Ici l'échelle est inversée : 100 = risque maximal, et la couleur tient compte de ce sens.",
        "work": ["Nommer à voix haute le frein probable (« j'ai l'impression que le prix vous gêne ») avant de reparler de signer.", "Traiter l'objection la plus ancienne non résolue en premier.", "Préparer le prochain appel avec la fiche de préparation générée à partir de ce rapport."],
    },
    "gaze_stability": {
        "how_short": "Mesuré sur la vidéo : part du temps où le regard reste face à l'écran.",
        "work_short": ['Repérez quand le regard décroche, relisez ce qui se disait', 'Détourner les yeux = souvent réfléchir, pas mentir', 'Ne concluez jamais sur ce signal seul'],
        "short": "La régularité du regard au fil de l'appel. Ce n'est PAS un détecteur de mensonge.",
        "how": "Mesuré directement sur la vidéo (pas par l'IA) : 100 moins la part du temps où le regard s'est détourné de l'écran, moyennée sur toutes les fenêtres où le visage est bien visible.",
        "work": ["Repérer les moments précis où le regard décroche et relire ce qui se disait.", "Un détournement avant une réponse = souvent de la réflexion, pas de la gêne : à confirmer par une question.", "Ne jamais conclure sur ce signal seul."],
    },
    "genuine_smile": {
        "how_short": 'Mesuré sur la vidéo : sourire + joues soulevées (le vrai sourire engage les yeux).',
        "work_short": ['Notez les sujets qui font sourire vraiment', "Sourire poli au prix ≠ accord : confirmez à l'oral", 'Score bas peut simplement dire « réservée »'],
        "short": "Sourire sincère (yeux inclus) plutôt que sourire de politesse.",
        "how": "Mesuré directement sur la vidéo : moyenne de l'intensité du sourire et du soulèvement des joues (le marqueur du sourire « de Duchenne », qui engage les yeux), sur toutes les fenêtres où le visage est bien visible.",
        "work": ["Noter à quels sujets le vrai sourire apparaît : ce sont les points d'adhésion.", "Un sourire poli au moment du prix ne vaut pas accord : poser une question de confirmation.", "Un score bas veut aussi dire « personne réservée », pas forcément « pas intéressée »."],
    },
    "gesture_stability": {
        "how_short": 'Mesuré sur la vidéo : les gestes restent-ils réguliers, sans pic isolé ?',
        "work_short": ['Regardez les pics : que disiez-vous à cet instant ?', 'Un pic sur une question = sujet sensible, allez-y en douceur', "Ne généralisez pas un pic à tout l'échange"],
        "short": "Les gestes restent-ils réguliers, ou y a-t-il des pics isolés ?",
        "how": "Mesuré directement sur la vidéo : variabilité relative des mouvements de mains au fil de l'appel (écart-type divisé par la moyenne, 100 = très régulier). Une personne qui bouge beaucoup mais de façon constante reste « stable » : on compare la personne à elle-même, pas à une norme.",
        "work": ["Regarder les pics isolés dans les moments clés : que se disait-il à cet instant ?", "Un pic de gestes sur une question précise est un sujet sensible à explorer en douceur.", "Ne pas généraliser un pic à tout l'échange."],
    },
}


def score_color(value: float, higher_is_better: bool = True, neutral: bool = False) -> str:
    if neutral:
        return NEUTRAL
    v = value if higher_is_better else 100 - value
    if v < 40:
        return RED
    if v < 70:
        return ORANGE
    return GREEN


def _zone_name(value: float, higher_is_better: bool, neutral: bool) -> str:
    if neutral:
        return "neutral"
    v = value if higher_is_better else 100 - value
    if v < 40:
        return "red"
    if v < 70:
        return "orange"
    return "green"


def _soft_tint(hex_color: str, white_ratio: float = 0.88) -> str:
    """Éclaircit une couleur en la mélangeant avec du blanc, en Python plutôt
    qu'avec color-mix() en CSS — weasyprint (rendu PDF) ne garantit pas le
    support des fonctions de couleur CSS récentes."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    mix = lambda c: round(c + (255 - c) * white_ratio)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


_ICONS = {
    "engagement": "🔥", "openness": "🤝", "buying_signal": "💰", "deal_risk": "⚠️",
    "gaze_stability": "👁️", "genuine_smile": "😊", "gesture_stability": "✋",
}
_LESSONS = {
    "engagement": "l2", "openness": "l2", "buying_signal": "l3", "deal_risk": "l4",
    "gaze_stability": "l5", "genuine_smile": "l5", "gesture_stability": "l5",
}
_VERDICTS = {"red": "À traiter", "orange": "À surveiller", "green": "Point fort", "neutral": "Indicatif"}
_GAUGE_R = 46
_GAUGE_CIRC = 2 * math.pi * _GAUGE_R


def axis_explanation(axis: RadarAxis) -> dict:
    info = _AXIS_INFO.get(axis.key, {"what": "", "why": "", "zones": {}})
    zone = _zone_name(axis.value, axis.higher_is_better, axis.neutral)
    color = score_color(axis.value, axis.higher_is_better, axis.neutral)
    return {
        "key": axis.key,
        "label": axis.label,
        "value": round(axis.value),
        "color": color,
        "color_soft": _soft_tint(color),
        "lesson": _LESSONS.get(axis.key, "l1"),
        "icon": _ICONS.get(axis.key, "•"),
        "verdict": _VERDICTS[zone],
        "gauge_r": _GAUGE_R,
        "gauge_dash": f"{_GAUGE_CIRC * max(0.0, min(axis.value, 100.0)) / 100:.1f} {_GAUGE_CIRC:.1f}",
        "short": _AXIS_EXTRA.get(axis.key, {}).get("short", ""),
        "how_short": _AXIS_EXTRA.get(axis.key, {}).get("how_short", ""),
        "work_short": _AXIS_EXTRA.get(axis.key, {}).get("work_short", []),
        "how": _AXIS_EXTRA.get(axis.key, {}).get("how", ""),
        "work": _AXIS_EXTRA.get(axis.key, {}).get("work", []),
        "what": info.get("what", ""),
        "why": info.get("why", ""),
        "comment": info.get("zones", {}).get(zone, ""),
    }


def build_radar(axes: list[RadarAxis], size: int = 420) -> dict:
    if not axes:
        return {}
    n = len(axes)
    cx = cy = size / 2
    r = size / 2 - 74
    angle_step = 2 * math.pi / n

    points, points_detail, axis_lines, labels = [], [], [], []
    for i, axis in enumerate(axes):
        angle = -math.pi / 2 + i * angle_step
        value = max(0.0, min(axis.value, 100.0))
        val_r = r * value / 100
        color = score_color(axis.value, axis.higher_is_better, axis.neutral)
        px, py = cx + val_r * math.cos(angle), cy + val_r * math.sin(angle)
        points.append(f"{px:.1f},{py:.1f}")
        points_detail.append({"x": round(px, 1), "y": round(py, 1), "color": color})

        ax, ay = cx + r * math.cos(angle), cy + r * math.sin(angle)
        axis_lines.append({"x1": round(cx, 1), "y1": round(cy, 1), "x2": round(ax, 1), "y2": round(ay, 1)})

        lx, ly = cx + (r + 40) * math.cos(angle), cy + (r + 40) * math.sin(angle)
        labels.append({
            "x": round(lx, 1), "y": round(ly, 1), "text": axis.label,
            "value": round(axis.value), "color": color,
            "anchor": _text_anchor(angle),
        })

    # Anneaux avec repère de valeur (25/50/75/100) affiché le long du tout
    # premier axe (vertical, vers le haut) — assez discret pour ne pas
    # surcharger, assez présent pour ancrer l'échelle du graphique.
    rings = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        rings.append({
            "r": round(r * frac, 1),
            "label_x": round(cx, 1),
            "label_y": round(cy - r * frac - 4, 1),
            "value": round(frac * 100),
        })

    # viewBox élargi horizontalement (pas juste "0 0 size size") : les labels
    # gauche/droite (text-anchor start/end) s'étendent au-delà du cercle
    # extérieur et seraient rognés par un viewBox carré strict.
    pad = 70
    return {
        "size": size, "cx": cx, "cy": cy, "r": r,
        "viewbox": f"{-pad} 0 {size + 2 * pad} {size}",
        "polygon_points": " ".join(points),
        "points_detail": points_detail,
        "axis_lines": axis_lines,
        "labels": labels,
        "rings": rings,
        "explanations": [axis_explanation(a) for a in axes],
    }


def _text_anchor(angle: float) -> str:
    cos_a = math.cos(angle)
    if cos_a > 0.3:
        return "start"
    if cos_a < -0.3:
        return "end"
    return "middle"
