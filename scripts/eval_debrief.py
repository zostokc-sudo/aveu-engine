"""Évaluation du débrief sur des appels SYNTHÉTIQUES dont on connaît les défauts.

Pourquoi : on n'a pas de vraies issues de deals, mais on peut au moins vérifier que le débrief VOIT ce qu'on y a mis :
un appel sans date de fin doit avoir une conclusion basse et « prochaine étape non fixée » ; un appel qui traite bien
l'objection doit mieux noter les objections. À relancer après chaque changement de prompt (workflow « Évaluer le débrief »).
Variables : GROQ_API_KEY."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from debrief import build_debrief  # noqa: E402
from transcribe import Segment  # noqa: E402

C, P = "other", "filmed"  # rôles : closer, prospect

BAD = [
    (C, "Bonjour Karim, alors je vous présente tout de suite notre programme, c'est un accompagnement complet de douze semaines avec des modules vidéo, des coachings de groupe et une communauté privée très active."),
    (P, "D'accord. Et ça se passe comment concrètement ?"),
    (C, "Alors il y a quarante modules, chaque module fait environ vingt minutes, vous avez aussi un accès à vie à la plateforme, et on a plus de trois cents élèves qui ont déjà suivi la formation avec de super résultats."),
    (P, "Oui mais moi j'ai surtout du mal à décrocher des rendez-vous qualifiés en ce moment, c'est mon vrai problème."),
    (C, "Ah oui, ça c'est normal, tout le monde a ce problème. Ce qui est génial c'est que dans le module trois on voit justement la prospection, et ensuite dans le module cinq on voit les scripts."),
    (P, "Et ça coûte combien ?"),
    (C, "Le programme est à quatre mille cinq cents euros. Voilà. Mais attendez, il y a aussi des bonus : un guide, des templates, et un accès au groupe privé pour toujours."),
    (P, "Hum, c'est un budget. Je dois en parler à mon associé avant de m'engager sur quoi que ce soit."),
    (C, "Ah d'accord, pas de souci. Moi je pense que c'est vraiment une super offre et que vous allez adorer."),
    (P, "Je comprends mais on a déjà investi dans un outil le trimestre dernier et ça n'a pas donné grand-chose."),
    (C, "Oui mais ça, ce n'est pas pareil, nous on est vraiment différents des autres. On est très sérieux."),
    (P, "Ok. Bon, on va réfléchir."),
    (C, "Très bien, pas de problème. Rappelez-moi quand vous voulez, je reste disponible. Bonne journée Karim !"),
    (P, "Merci, bonne journée."),
]

GOOD = [
    (C, "Bonjour Karim, merci d'avoir pris ce créneau. On a trente minutes. Voici comment je vous propose de faire : je vous pose des questions sur votre situation, on voit si je peux vous aider, et si c'est le cas on décide ensemble de la suite avant de raccrocher. Ça vous va ?"),
    (P, "Oui, très bien."),
    (C, "Parfait. Pour commencer, qu'est-ce qui vous a poussé à demander cet échange cette semaine ?"),
    (P, "J'ai du mal à décrocher des rendez-vous qualifiés, et mon chiffre d'affaires stagne depuis six mois."),
    (C, "Stagne à combien aujourd'hui, et vous visiez quoi ?"),
    (P, "On est à vingt mille par mois, je voulais atteindre quarante avant la fin de l'année."),
    (C, "Et qu'est-ce que vous avez déjà essayé pour y arriver ?"),
    (P, "J'ai testé un outil de prospection et une agence, ça n'a pas donné grand-chose."),
    (C, "D'accord. Qu'est-ce qui se passe si rien ne change dans six mois ?"),
    (P, "Honnêtement, je ne pourrai pas recruter, et je commence à douter de mon projet."),
    (C, "Pourquoi maintenant, et pas dans six mois ?"),
    (P, "Parce que mon associé et moi avons fixé une échéance en décembre. Après, on revoit tout."),
    (C, "Si je résume : vingt mille aujourd'hui, quarante visés, deux essais sans résultat, et une échéance en décembre qui décide de la suite. C'est bien ça ?"),
    (P, "Exactement."),
    (C, "Dans ce cas voici ce que je vous propose : un accompagnement de douze semaines centré sur la prospection de rendez-vous qualifiés. L'investissement est de quatre mille cinq cents euros. Qu'en pensez-vous ?"),
    (P, "C'est un budget. Je dois en parler à mon associé."),
    (C, "Je comprends. Quand vous dites que c'est un budget, c'est par rapport à quoi exactement : le montant, ou la certitude d'un résultat ?"),
    (P, "Plutôt la certitude. L'outil précédent m'a refroidi."),
    (C, "C'est légitime. Ce qui était différent dans votre outil précédent : personne ne regardait vos vrais appels. Ici, chaque semaine, on analyse vos appels ensemble. Est-ce que ça répond à votre inquiétude ?"),
    (P, "Oui, ça change la donne."),
    (C, "Et votre associé, qu'est-ce qui le convaincrait ? On peut faire un point à trois demain."),
    (P, "Un point à trois, oui, ce serait bien. Demain en fin de matinée ?"),
    (C, "Parfait : demain, onze heures, nous trois. Je vous envoie l'invitation maintenant, et je vous réserve la place d'ici là. On se retrouve demain à onze heures."),
    (P, "Très bien, à demain."),
]

MID = [
    (C, "Bonjour Karim, alors, qu'est-ce qui vous amène aujourd'hui ?"),
    (P, "J'ai du mal à décrocher des rendez-vous qualifiés."),
    (C, "D'accord, et depuis combien de temps ?"),
    (P, "Environ six mois, mon chiffre d'affaires stagne."),
    (C, "Je vois. Notre accompagnement dure douze semaines et il est centré sur la prospection. Il coûte quatre mille cinq cents euros."),
    (P, "C'est un budget. Je dois en parler à mon associé."),
    (C, "Bien sûr. Est-ce que vous pensez que ça pourrait vous aider ?"),
    (P, "Peut-être, je ne sais pas trop."),
    (C, "Écoutez, réfléchissez-y avec votre associé et on se rappelle dans la semaine, ça vous va ?"),
    (P, "Oui, dans la semaine, très bien."),
]


def segments(lines):
    out, t = [], 0.0
    for who, text in lines:
        dur = max(6.0, len(text) / 14.0)
        out.append(Segment(text, t, t + dur, [], speaker=who))
        t += dur + 1.5
    return out


def padded(lines, repeat):
    """Allonge un appel court pour franchir les seuils de longueur minimale du débrief (texte inchangé en substance)."""
    return lines * repeat if repeat > 1 else lines


CASES = {"mauvais": padded(BAD, 1), "bon": padded(GOOD, 1), "moyen": padded(MID, 3)}


def main() -> int:
    results = {}
    for name, lines in CASES.items():
        print(f"\n=== Appel « {name} » ===")
        d = build_debrief(segments(lines), roles_inferred=False)
        if d is None:
            print("Débrief non généré (texte jugé trop pauvre)")
            continue
        results[name] = d
        print("scores :", {c["key"]: c["score"] for c in d["scorecard"]}, "| étape suivante fixée :", d["next_step_secured"])
        print("priorités :", d["priorities"])
        time.sleep(60)  # plafond gratuit Groq

    ok = True

    def check(label, cond):
        nonlocal ok
        print(("✅" if cond else "❌"), label)
        ok &= bool(cond)

    s = lambda n, k: next(c["score"] for c in results[n]["scorecard"] if c["key"] == k)
    if "mauvais" in results and "bon" in results:
        check("la conclusion du bon appel est supérieure d'au moins 25 points à celle du mauvais", s("bon", "closing") - s("mauvais", "closing") >= 25)
        check("le bon appel gère mieux les objections (+20 points)", s("bon", "objections") - s("mauvais", "objections") >= 20)
        check("la découverte du bon appel est meilleure (+20 points)", s("bon", "discovery") - s("mauvais", "discovery") >= 20)
        check("« prochaine étape fixée » : non pour le mauvais, oui pour le bon", (not results["mauvais"]["next_step_secured"]) and results["bon"]["next_step_secured"])
        check("le mauvais appel est noté sous 50 en conclusion", s("mauvais", "closing") < 50)
    else:
        check("les deux appels de référence ont produit un débrief", False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
