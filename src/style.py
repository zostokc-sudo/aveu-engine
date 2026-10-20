"""Style de communication du PROSPECT (4 couleurs) : une grille pour s'adapter, jamais un diagnostic.

Le modèle « DISC » (rouge, jaune, vert, bleu) est populaire en vente mais sa validité scientifique est faible
(fiabilité et pouvoir prédictif contestés). Aveu l'utilise donc comme repère de conversation : on décrit une
TENDANCE OBSERVÉE dans cet échange (ce dont le prospect parle, son rythme, ce qu'il demande), avec des citations
réelles, une confiance au mieux « moyenne », et des conseils d'adaptation fixes. Rien n'est tiré du visage."""

from typing import Literal

from pydantic import BaseModel, Field

from debrief import _call, locate_quote

_Color = Literal["rouge", "jaune", "vert", "bleu", "indetermine"]


class StyleEvidence(BaseModel):
    t_sec: int = Field(description="seconde de la citation ([mm:ss] de la transcription)")
    quote: str = Field(description="citation du PROSPECT copiée mot pour mot, 160 caractères maximum")


class StyleProfile(BaseModel):
    dominant: _Color = Field(description="couleur dominante dans cet échange, ou 'indetermine' si le texte ne permet pas de trancher")
    secondary: _Color = Field(description="seconde couleur, ou 'indetermine'")
    confidence: Literal["faible", "moyenne"] = Field(description="jamais plus que 'moyenne' : un seul échange ne suffit pas")
    evidence: list[StyleEvidence] = Field(description="1 à 3 citations qui justifient le choix")
    why: str = Field(description="deux phrases maximum : quels comportements de langage observés, sans jugement de personnalité")


STYLES = {
    "rouge": {"emoji": "🔴", "label": "Rouge — direct, orienté résultat",
              "traits": "Va droit au but, décide vite, parle de résultats et de délais, supporte mal les détours.",
              "tips": ["Soyez bref : le résultat d'abord, le détail ensuite.", "Donnez-lui le contrôle du choix (« deux options, laquelle préférez-vous ? »).", "Parlez délai et retour concret, pas processus."]},
    "jaune": {"emoji": "🟡", "label": "Jaune — expressif, enthousiaste",
              "traits": "Raconte, aime la vision et les idées, réagit à l'énergie et à la reconnaissance, se disperse parfois.",
              "tips": ["Mettez de l'énergie, racontez un cas client qui lui ressemble.", "Faites-le parler de sa vision, puis ramenez-le à une étape précise.", "Fixez une date d'engagement : il adhère vite mais peut oublier."]},
    "vert": {"emoji": "🟢", "label": "Vert — stable, prudent",
             "traits": "Cherche la sécurité et l'harmonie, décide en consultant, n'aime pas être bousculé.",
             "tips": ["Gardez un rythme calme, ne le pressez pas.", "Rassurez : garanties, témoignages, accompagnement.", "Proposez d'impliquer son entourage ou son associé dans la décision."]},
    "bleu": {"emoji": "🔵", "label": "Bleu — analytique, précis",
             "traits": "Veut des faits, des chiffres et de la cohérence, pose des questions de détail, vérifie avant de s'engager.",
             "tips": ["Apportez des chiffres, des comparaisons et un processus clair.", "Répondez précisément, sans exagérer : il repère les approximations.", "Laissez-lui un délai de vérification, avec une date de retour fixée."]},
}

_SYSTEM = """\
Tu lis ce que dit le PROSPECT dans un appel de vente et tu décris sa TENDANCE de communication dans cet échange, selon 4 \
couleurs : rouge (direct, résultat, rapidité), jaune (expressif, enthousiaste, vision, histoires), vert (prudent, sécurité, \
harmonie, consulte son entourage), bleu (analytique, chiffres, détails, vérification).

RÈGLES : tu décris des comportements de langage observés, jamais la personnalité ni l'état émotionnel ; tu choisis \
'indetermine' si le texte ne permet pas de trancher ; la confiance n'est jamais supérieure à 'moyenne' ; les citations sont \
copiées MOT POUR MOT des lignes du prospect. Réponds en français."""


def style_text_available(segments, roles_known: bool) -> bool:
    return sum(len(s.text) for s in segments if s.text.strip() and (not roles_known or s.speaker == 'filmed')) >= 500


def build_style(segments, roles_known: bool) -> dict | None:
    mine = [s for s in segments if s.text.strip() and (not roles_known or s.speaker == "filmed")]
    if sum(len(s.text) for s in mine) < 500:
        print("Style du prospect : texte du prospect insuffisant.")
        return None
    if len(mine) > 40:  # échantillon réparti sur TOUT l'appel (et pas seulement son début)
        sample = [mine[int(i * len(mine) / 40)] for i in range(40)]
    else:
        sample = mine
    lines = "\n".join(f"[{int(s.start) // 60:02d}:{int(s.start) % 60:02d}] {s.text.strip()}" for s in sample)[:6500]
    who = "Lignes du prospect" if roles_known else "Transcription (le prospect est celui qui répond aux questions et exprime un besoin)"
    result = _call(_SYSTEM, f"{who} :\n\n{lines}", StyleProfile, 900)
    if result.dominant == "indetermine":
        print("Style du prospect : le texte ne permet pas de trancher.")
        return None
    evidence = []
    for e in result.evidence:
        at = locate_quote(e.quote, segments, speaker="filmed" if roles_known else None)
        if at is not None:
            evidence.append({"at": f"{int(at) // 60:02d}:{int(at) % 60:02d}", "quote": e.quote})
    evidence = evidence[:3]
    if not evidence:
        print("Style du prospect : aucune citation retrouvée dans la transcription, profil écarté.")
        return None
    info = STYLES[result.dominant]
    second = STYLES.get(result.secondary)
    return {
        "dominant": result.dominant, "emoji": info["emoji"], "label": info["label"], "traits": info["traits"], "tips": info["tips"],
        "secondary_label": second["label"] if second and result.secondary != result.dominant else "",
        "confidence": result.confidence, "why": result.why,
        "evidence": evidence,
    }
