"""Bilan du DISCOURS (texte) : qui parle combien, qui pose les questions, qui monopolise la parole.

Ce module ne lit que les mots et les temps de parole — aucune émotion n'est inférée, donc rien ici
ne touche à l'interdiction de l'AI Act (art. 5) : on peut l'appliquer au closer lui-même pour son
propre débrief. Il ne fonctionne que si les voix ont été séparées (diarize.py) ; sinon il renvoie None."""

_QUESTION_WORDS = ("comment ", "pourquoi ", "qu'est-ce", "est-ce que", "combien ", "quand ", "what ", "how ", "why ")


def _is_question(text: str) -> bool:
    t = text.strip().lower()
    return t.endswith("?") or any(t.startswith(w) for w in _QUESTION_WORDS)


def transcript_lines(segments) -> list[dict]:
    """Transcription complète, étiquetée quand les voix sont séparées."""
    names = {"filmed": "Prospect (à l'écran)", "other": "Interlocuteur"}
    return [
        {"t": int(s.start), "who": names.get(s.speaker, ""), "text": s.text}
        for s in segments if s.text.strip()
    ]


def conversation_metrics(segments) -> dict | None:
    labelled = [s for s in segments if s.speaker in ("filmed", "other")]
    if len(labelled) < 4 or not any(s.speaker == "other" for s in labelled) or not any(s.speaker == "filmed" for s in labelled):
        return None
    talk = {"filmed": 0.0, "other": 0.0}
    questions = {"filmed": 0, "other": 0}
    longest = {"filmed": 0.0, "other": 0.0}
    run_speaker, run_len = None, 0.0
    for s in labelled:
        d = max(0.0, s.end - s.start)
        talk[s.speaker] += d
        questions[s.speaker] += 1 if _is_question(s.text) else 0
        if s.speaker == run_speaker:
            run_len += d
        else:
            run_speaker, run_len = s.speaker, d
        longest[s.speaker] = max(longest[s.speaker], run_len)
    total = talk["filmed"] + talk["other"]
    if total <= 0:
        return None
    other_share = round(100 * talk["other"] / total)
    return {
        "other_share": other_share,
        "filmed_share": 100 - other_share,
        "other_questions": questions["other"],
        "filmed_questions": questions["filmed"],
        "other_longest_sec": round(longest["other"]),
        "advice": (
            "Vous avez parlé plus de la moitié du temps : donnez-lui plus d'espace avec des questions ouvertes."
            if other_share > 55 else
            "Bon équilibre : vous laissez le prospect parler (les meilleurs closers parlent un peu moins de la moitié du temps)."
        ),
        "monologue_warning": longest["other"] > 90,
    }
