"""Fiche « Mes coachings » : version compacte et structurée d'un débrief, lue par l'espace personnel du site.

Rien n'est généré en plus par l'IA : tout vient du débrief déjà vérifié (citations retrouvées dans la transcription)."""

import re
from datetime import date

# clés du débrief -> clés de l'espace personnel (landing/app-data.js)
_APP_KEYS = {"opening": "open", "listening": "listen", "discovery": "discover", "proposal": "pitch", "objections": "objec", "closing": "close"}
_ORDER = ["opening", "discovery", "listening", "objections", "proposal", "closing"]


def _sentences(text: str, limit: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts[:limit]


def build_coaching(debrief: dict, extras: dict | None, subject: str, duration_min: float,
                   source_url: str | None = None) -> dict:
    by_key = {d["key"]: d for d in debrief["scorecard"]}
    scores = {_APP_KEYS[k]: d["score"] for k, d in by_key.items() if k in _APP_KEYS}
    # résumé en 10 lignes maximum : 3 phrases d'ensemble, une ligne par étape, la fin d'appel
    resume = _sentences(debrief["summary"], 3)
    resume += [by_key[k]["verdict"].strip() for k in _ORDER if k in by_key and by_key[k]["verdict"].strip()]
    if debrief.get("next_step_detail"):
        resume.append(debrief["next_step_detail"].strip())
    forces = [s["why"] for s in debrief["strengths"] if s.get("why")]
    if not forces and by_key:
        best = max(by_key.values(), key=lambda d: d["score"])
        forces = [f"{best['label']} : {best['verdict']}"]
    moments = [{"t": s["at"], "k": "good", "quote": s["quote"], "why": s["why"], "tip": "À garder : refaites-le au prochain appel."}
               for s in debrief["strengths"]]
    moments += [{"t": m["at"], "k": "work", "quote": m["prospect_said"], "why": m["closer_did"], "tip": m["better_move"]}
                for m in debrief["missed"]]
    moments.sort(key=lambda m: m["t"])
    style = (extras or {}).get("style")
    prospect = None
    if style:
        prospect = {"couleur": style["dominant"], "traits": [t["key"] for t in style.get("traits", [])], "note": style["why"]}
    return {
        "titre": subject, "date": date.today().isoformat(), "duree": f"{round(duration_min)} min", "theme": "",
        "issue": "Inconnue", "video": source_url, "prospect": prospect, "scores": scores,
        "resume": resume[:10], "forces": forces[:4], "axes": list(debrief["priorities"]), "moments": moments,
        "next_step_secured": debrief["next_step_secured"],
    }
