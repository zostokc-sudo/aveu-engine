"""Contrôles de qualité automatiques sur un rapport généré — pour objectiver
si un problème est corrigé plutôt que de se fier à une relecture à l'oeil.

Né d'un défaut observé : le modèle local recyclait souvent la même
interprétation d'un moment clé à l'autre. Un prompt peut réduire ce risque,
mais ne le garantit jamais — ce module le détecte après coup, pour chaque
modèle (local ou production), sans dépendre de la discipline du prompt seul.
"""

from difflib import SequenceMatcher

from schema import BehaviorReport, PreCallBrief

# Deux interprétations avec un ratio de similarité au-dessus de ce seuil sont
# considérées comme des redites plutôt que deux lectures réellement distinctes.
_SIMILARITY_THRESHOLD = 0.6


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# behavioral_signal doit être un court libellé, ou plusieurs libellés courts
# joints par "+" ("bouche fermée + hausse du ton + mouvements de tête") —
# légitime même long au total. Ce qui est suspect, c'est un SEGMENT individuel
# trop long : signe d'une phrase/clause complète déguisée en libellé, trouvé
# en pratique quand le modèle local y écrivait une interprétation en toutes
# lettres ("La personne sourit et son ton monte, montrant un engagement...").
# On juge donc chaque segment séparé par "+", pas la longueur totale.
_MAX_WORDS_PER_SIGNAL_SEGMENT = 6


def detect_verbose_signal_labels(report: BehaviorReport) -> list[str]:
    """Repère les behavioral_signal dont un segment (séparé par "+") est trop
    long pour être un libellé — signe qu'une phrase complète s'est glissée
    dans ce champ plutôt qu'une liste de courts signaux."""
    warnings = []
    for i, moment in enumerate(report.key_moments):
        segments = [s.strip() for s in moment.behavioral_signal.split("+")]
        longest = max(segments, key=lambda s: len(s.split())) if segments else ""
        if len(longest.split()) > _MAX_WORDS_PER_SIGNAL_SEGMENT:
            warnings.append(
                f"key_moments[{i}].behavioral_signal contient un segment trop "
                f"long (« {longest[:50]}... ») — devrait être un court libellé, "
                "pas une phrase ; l'explication appartient à interpretation."
            )
    return warnings


def _find_repetitive_pairs(texts: list[str]) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = _similarity(texts[i], texts[j])
            if ratio > _SIMILARITY_THRESHOLD:
                pairs.append((i, j, round(ratio, 2)))
    return pairs


def detect_repetitive_moments(report: BehaviorReport) -> list[tuple[int, int, float]]:
    """Retourne les paires (index_i, index_j, ratio) de key_moments dont les
    interprétations se ressemblent trop pour apporter une lecture distincte."""
    return _find_repetitive_pairs([m.interpretation for m in report.key_moments])


# Fraction minimale des mots de la citation qui doivent apparaître dans le
# texte réellement transcrit autour de ce moment. Volontairement bas : le
# schéma autorise "verbatim ou résumé court", donc un résumé légitime peut
# reformuler — ce seuil n'attrape que les citations qui ne correspondent à
# RIEN de ce qui a été dit à ce moment (invention pure), pas une paraphrase.
_QUOTE_GROUNDING_THRESHOLD = 0.2
_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "au", "aux",
    "ce", "ces", "cette", "que", "qui", "pas", "on", "il", "elle", "je", "tu",
    "vous", "nous", "ils", "elles", "en", "d", "l", "c", "j", "n", "s", "y",
}


def verify_quotes_grounded(report: BehaviorReport, timeline: list[dict]) -> list[str]:
    """Vérifie que chaque citation d'un key_moment correspond à quelque chose
    de réellement transcrit dans la fenêtre temporelle correspondante — filet
    de sécurité contre une citation inventée par le LLM plutôt qu'extraite des
    données fournies."""
    warnings = []
    for i, moment in enumerate(report.key_moments):
        overlapping_text = " ".join(
            w.get("spoken_text", "") for w in timeline
            if w["start_sec"] < moment.end_sec and w["end_sec"] > moment.start_sec
        ).lower()
        quote_words = {w for w in moment.quote.lower().split() if w not in _STOPWORDS and len(w) > 2}
        if not quote_words:
            continue
        overlap_words = set(overlapping_text.split())
        match_ratio = len(quote_words & overlap_words) / len(quote_words)
        if match_ratio < _QUOTE_GROUNDING_THRESHOLD:
            warnings.append(
                f"key_moments[{i}] : la citation « {moment.quote[:60]}... » ne "
                "correspond à presque rien du texte transcrit à ce moment — "
                "possible invention, à vérifier."
            )
    return warnings


# Signaux biométriques que cet outil ne mesure jamais — trouvés en pratique
# dans une sortie du modèle local ("fréquence cardiaque légèrement accélérée")
# alors qu'aucune estimation de rythme cardiaque n'existe nulle part dans le
# pipeline. Une fabrication plausible mais fausse sur ce que l'outil mesure
# réellement est un vrai risque de confiance pour un livrable vendu cher.
_FABRICATED_SIGNAL_TERMS = [
    "fréquence cardiaque", "rythme cardiaque", "pouls", "battements de cœur",
    "transpiration", "sueur", "dilatation des pupilles", "pupille",
    "conductance de la peau", "température corporelle", "rougissement",
]


def detect_fabricated_signals(report: BehaviorReport) -> list[str]:
    """Repère les mentions de signaux biométriques que l'outil ne mesure
    jamais (voir _FABRICATED_SIGNAL_TERMS) dans behavioral_signal ou
    interpretation — signe que le LLM a comblé avec sa connaissance générale
    plutôt que de s'en tenir aux données fournies."""
    warnings = []
    for i, moment in enumerate(report.key_moments):
        text = f"{moment.behavioral_signal} {moment.interpretation}".lower()
        for term in _FABRICATED_SIGNAL_TERMS:
            if term in text:
                warnings.append(
                    f"key_moments[{i}] mentionne « {term} » — signal jamais mesuré "
                    "par cet outil, probable fabrication à corriger."
                )
    return warnings


# Vocabulaire de diagnostic clinique/définitif que le prompt interdit
# explicitement (RÈGLE CENTRALE : jamais de diagnostic de stress ; prudence
# DePaulo et al. 2003 sur le regard comme indice de mensonge) mais qu'un
# modèle — surtout le modèle local, moins docile sur les consignes fines —
# peut réintroduire par réflexe de langage courant. Trouvé en pratique : le
# modèle local respecte souvent la règle sur les key_moments mais peut encore
# la relâcher dans coaching_recommendations ou executive_summary, deux champs
# que les premiers contrôles ne couvraient pas.
_DIAGNOSTIC_LANGUAGE_TERMS = [
    "stressé", "stressée", "anxieux", "anxieuse", "angoissé", "angoissée",
    "paniqué", "paniquée", "menteur", "menteuse", "mensonge", "malhonnête",
    "malhonnete",
]


def detect_diagnostic_language(report: BehaviorReport) -> list[str]:
    """Repère un diagnostic clinique/définitif interdit (stress, mensonge, etc.)
    n'importe où dans le rapport — pas seulement dans behavioral_signal et
    interpretation, mais aussi executive_summary et coaching_recommendations,
    où la même règle s'applique mais n'était pas vérifiée jusqu'ici."""
    warnings = []
    fields = [("executive_summary", report.executive_summary), ("closing_verdict", report.closing_verdict)]
    fields += [(f"coaching_recommendations[{i}]", r) for i, r in enumerate(report.coaching_recommendations)]
    fields += [
        (f"key_moments[{i}]", f"{m.behavioral_signal} {m.interpretation}")
        for i, m in enumerate(report.key_moments)
    ]
    for field_name, text in fields:
        low = text.lower()
        for term in _DIAGNOSTIC_LANGUAGE_TERMS:
            if term in low:
                warnings.append(
                    f"{field_name} contient « {term} » — diagnostic clinique/"
                    "définitif interdit par le prompt, à reformuler en pattern "
                    "observable."
                )
    return warnings


def quality_warnings(report: BehaviorReport, timeline: list[dict] | None = None) -> list[str]:
    """Avertissements lisibles — utile pour du QA manuel ou un log serveur,
    jamais pour bloquer la génération : un rapport imparfait vaut mieux
    qu'aucun rapport."""
    warnings = []
    repetitive = detect_repetitive_moments(report)
    for i, j, ratio in repetitive:
        warnings.append(
            f"key_moments[{i}] et key_moments[{j}] ont des interprétations "
            f"trop proches (similarité {ratio}) — manque de variété."
        )
    if len(report.key_moments) < 3:
        warnings.append(f"Seulement {len(report.key_moments)} moment(s) clé(s) — rapport peu étoffé.")
    warnings.extend(detect_fabricated_signals(report))
    warnings.extend(detect_verbose_signal_labels(report))
    warnings.extend(detect_diagnostic_language(report))
    if timeline is not None:
        warnings.extend(verify_quotes_grounded(report, timeline))
    return warnings


def precall_quality_warnings(brief: PreCallBrief) -> list[str]:
    """Mêmes principes que quality_warnings(), appliqués à la fiche de
    préparation : répétitivité entre les entrées d'une même liste (deux
    objections probables qui disent la même chose n'apportent rien de plus
    qu'une seule), et signaux biométriques fabriqués dans n'importe quel champ
    texte."""
    warnings = []
    for field_name, items in (
        ("likely_objections", brief.likely_objections),
        ("prep_actions", brief.prep_actions),
        ("watch_for", brief.watch_for),
        ("verification_questions", brief.verification_questions),
    ):
        for i, j, ratio in _find_repetitive_pairs(items):
            warnings.append(
                f"{field_name}[{i}] et {field_name}[{j}] se ressemblent trop "
                f"(similarité {ratio}) — manque de variété."
            )

    all_text = " ".join(
        [brief.context_recap, brief.opening_move, *brief.likely_objections,
         *brief.prep_actions, *brief.watch_for, *brief.verification_questions]
    ).lower()
    for term in _FABRICATED_SIGNAL_TERMS:
        if term in all_text:
            warnings.append(f"La fiche mentionne « {term} » — signal jamais mesuré par cet outil.")
    for term in _DIAGNOSTIC_LANGUAGE_TERMS:
        if term in all_text:
            warnings.append(
                f"La fiche mentionne « {term} » — diagnostic clinique/définitif "
                "interdit par le prompt, à reformuler en pattern observable."
            )
    return warnings
