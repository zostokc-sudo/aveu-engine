"""Voix et vocabulaire du CLOSER : des faits de langage, jamais des émotions.

Débit (mots par minute), mots de remplissage, vocabulaire « mou » (diminutifs, précautions, formules qui
n'engagent pas) et niveau de challenge des questions. Tout est compté sur le texte transcrit et les temps de
parole du closer : aucune lecture de l'état émotionnel, donc rien qui touche à l'interdiction de l'AI Act (art. 5).

Ne fonctionne que si les voix sont séparées (le closer = segments étiquetés 'other')."""

import re

_FILLERS_FR = ["euh", "heu", "ben", "bah", "genre", "du coup", "en fait", "tu vois", "vous voyez", "quoi", "hein", "voilà", "enfin bref", "bref"]
_FILLERS_EN = ["um", "uh", "you know", "like", "i mean", "kind of", "sort of"]
_SOFT = [
    "petit", "petite", "petits", "petites", "juste", "un peu", "peut-être", "peut être", "je pense que", "je crois que",
    "en quelque sorte", "si ça vous dit", "si ça te dit", "n'hésitez pas", "n'hésite pas", "désolé", "je voulais", "j'aurais voulu",
    "essayer de", "j'espère", "normalement", "a priori", "pas grand-chose",
]
_CHALLENGE = [
    "pourquoi", "qu'est-ce qui se passe si", "et si vous ne", "et si tu ne", "qu'est-ce qui vous empêche", "qu'est-ce qui t'empêche",
    "combien ça vous coûte", "combien ça te coûte", "qu'est-ce que vous avez déjà essayé", "qu'est-ce que tu as déjà essayé",
    "concrètement", "vraiment", "pourquoi maintenant", "dans 6 mois", "si rien ne change", "what happens if", "why haven't", "what's stopping",
]
_QUESTION_START = ("comment ", "pourquoi ", "qu'est-ce", "est-ce que", "combien ", "quand ", "quel", "quelle", "où ", "what ", "how ", "why ", "when ", "do you", "are you", "is it")


def _count(text: str, phrases: list[str]) -> dict[str, int]:
    t = " " + re.sub(r"[^\w'àâçéèêëîïôûùüÿœ -]", " ", text.lower()) + " "
    found = {}
    for p in phrases:
        n = len(re.findall(r"(?<![\w'])" + re.escape(p) + r"(?![\w'])", t))
        if n:
            found[p] = n
    return found


def _is_question(text: str) -> bool:
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(_QUESTION_START)


def _mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def delivery_metrics(segments) -> dict | None:
    mine = [s for s in segments if getattr(s, "speaker", None) == "other" and s.text.strip()]
    if len(mine) < 4:
        return None
    talk_sec = sum(max(0.0, s.end - s.start) for s in mine)
    text = " ".join(s.text for s in mine)
    words = len(re.findall(r"[\w'àâçéèêëîïôûùüÿœ-]+", text))
    if talk_sec < 30 or words < 80:
        return None
    minutes = talk_sec / 60
    wpm = round(words / minutes)

    english = len(_count(text, ["the", "and", "you"])) >= 2 and not _count(text, ["le", "la", "et", "vous", "je"])
    fillers = _count(text, _FILLERS_EN if english else _FILLERS_FR)
    soft = _count(text, _SOFT)
    filler_total, soft_total = sum(fillers.values()), sum(soft.values())

    questions = [s for s in mine if _is_question(s.text)]
    challenging = [s for s in questions if _count(s.text, _CHALLENGE)]

    # extraits réels (vérifiables) du vocabulaire « mou », avec leur minute
    examples = []
    for s in mine:
        hit = _count(s.text, _SOFT)
        if hit and len(examples) < 3:
            examples.append({"at": _mmss(s.start), "text": s.text.strip()[:140], "words": sorted(hit)[:3]})

    long_pauses = 0
    for s in mine:
        ws = getattr(s, "words", None) or []
        long_pauses += sum(1 for a, b in zip(ws, ws[1:]) if b.start - a.end > 1.5)

    per_min = filler_total / minutes
    soft_per_10 = soft_total / (minutes / 10)
    challenge_pct = round(100 * len(challenging) / len(questions)) if questions else 0
    return {
        "wpm": wpm,
        "wpm_label": "un peu lent" if wpm < 110 else "naturel" if wpm <= 175 else "rapide",
        "wpm_advice": (
            "Un débit lent peut sembler hésitant : accélérez légèrement et marquez vos pauses aux moments clés."
            if wpm < 110 else
            "Bon rythme : ni pressé ni traînant. Gardez des pauses après vos questions importantes."
            if wpm <= 175 else
            "Débit rapide : le prospect peut décrocher. Ralentissez sur le prix et sur vos questions clés."
        ),
        "fillers_per_min": round(per_min, 1),
        "fillers_top": sorted(fillers.items(), key=lambda kv: -kv[1])[:3],
        "fillers_label": "rares" if per_min < 2 else "présents" if per_min <= 5 else "nombreux",
        "soft_total": soft_total,
        "soft_per_10min": round(soft_per_10, 1),
        "soft_top": sorted(soft.items(), key=lambda kv: -kv[1])[:4],
        "soft_label": "percutant" if soft_per_10 < 8 else "à durcir" if soft_per_10 <= 20 else "trop mou",
        "soft_advice": (
            "Vocabulaire assez affirmé. Continuez à dire « je vous recommande » plutôt que « je pense que »."
            if soft_per_10 < 8 else
            "Plusieurs mots qui adoucissent trop (« juste », « un peu », « petit »). Remplacez-les par des formulations nettes."
            if soft_per_10 <= 20 else
            "Beaucoup de mots qui affaiblissent votre propos : le prospect y lit de l'hésitation. Visez des phrases courtes et affirmées."
        ),
        "soft_examples": examples,
        "questions": len(questions),
        "challenge_pct": challenge_pct,
        "challenge_label": "challengeant" if challenge_pct >= 30 else "à muscler",
        "challenge_advice": (
            "Vous bousculez le prospect avec de bonnes questions : il s'implique."
            if challenge_pct >= 30 else
            "Peu de questions qui challengent. Osez : « Qu'est-ce qui se passe si rien ne change dans 6 mois ? », « Pourquoi maintenant ? »."
        ),
        "long_pauses": long_pauses,
        "talk_minutes": round(minutes, 1),
    }
