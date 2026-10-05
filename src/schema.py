"""Schéma structuré du rapport d'analyse comportementale."""

from pydantic import BaseModel, Field


class Scores(BaseModel):
    engagement: int = Field(description="0-100, niveau d'engagement global de la personne")
    activation: int = Field(description=(
        "0-100, intensité de l'activation physiologique observée (rythme, gestes, "
        "regard, respiration). ATTENTION : une activation haute n'est PAS un "
        "diagnostic de stress — elle peut tout autant traduire de l'enthousiasme, "
        "l'envie de convaincre, ou une forte implication. Ce chiffre mesure "
        "l'intensité, jamais sa cause ; la cause se lit dans le contexte verbal "
        "de chaque key_moment, jamais depuis ce score seul. Ce découplage n'est "
        "pas arbitraire : il suit le modèle circomplexe de l'affect (Russell, "
        "1980), qui établit que l'arousal (intensité de l'activation) et la "
        "valence (positive ou négative) sont deux dimensions INDÉPENDANTES d'un "
        "état affectif — mesurer l'une ne renseigne pas sur l'autre. Ce score est "
        "une mesure d'arousal pur ; ne jamais lui faire porter une valence "
        "implicite (haut = mauvais signe)."
    ))
    openness: int = Field(description="0-100, ouverture/confiance (posture, sourire, contact visuel)")
    buying_signal: int = Field(description="0-100, force des signaux d'achat/d'accord")
    deal_risk: int = Field(description="0-100, risque que l'affaire échoue ou traîne")


class KeyMoment(BaseModel):
    start_sec: float = Field(description="début du moment clé, en secondes")
    end_sec: float = Field(description="fin du moment clé, en secondes")
    quote: str = Field(description="ce qui a été dit à ce moment, verbatim ou résumé court — ATTENTION : la transcription ne distingue pas les locuteurs, ce peut être l'un ou l'autre camp de la conversation, ne jamais affirmer que c'est forcément l'interlocuteur externe qui parle si le contexte ne le confirme pas clairement")
    behavioral_signal: str = Field(description=(
        "COURT LIBELLÉ des signaux observés (ex: 'sourcils froncés + regard "
        "fuyant'), 5 mots maximum, jamais une phrase complète — la partie "
        "explicative (POURQUOI, ce que ça signifie) va dans interpretation, pas "
        "ici. Ce champ est affiché en majuscules comme une étiquette courte : "
        "'La personne sourit et son ton monte, montrant de l'engagement' n'est "
        "PAS un libellé valide (c'est une interprétation déguisée), "
        "'sourire + hausse du ton' l'est. Limite-toi STRICTEMENT aux "
        "signaux fournis dans les données (visage, "
        "regard, sourcils, mâchoire, mains/gestes, posture, voix/hauteur/énergie). "
        "N'invente JAMAIS un signal qui n'est pas dans les données fournies — en "
        "particulier, ne mentionne jamais la fréquence cardiaque, le pouls, la "
        "transpiration, la dilatation des pupilles, la conductance de la peau, ou "
        "tout autre signal biométrique : rien de tout cela n'est mesuré par cet "
        "outil, même si ce sont des signes de stress plausibles en général."
    ))
    interpretation: str = Field(description=(
        "l'interprétation de ce signal dans son contexte. RÈGLE : ne jamais poser "
        "un diagnostic unique et définitif (ex: 'il est stressé'). Une activation "
        "physiologique a plusieurs causes plausibles (stress, excitation, désir de "
        "convaincre, forte implication) — nomme la ou les lectures les plus "
        "probables compte tenu de ce qui est dit à ce moment, formule-les de façon "
        "à ce que la personne concernée se reconnaisse ('semble vouloir appuyer son "
        "propos avec force' plutôt que 'est stressé'), et n'affirme que ce que le "
        "verbal permet de trancher. VARIÉTÉ OBLIGATOIRE : ne réutilise jamais la "
        "même phrase-type ou la même explication pour deux moments différents, "
        "même si les signaux bruts se ressemblent — si deux moments ont une lecture "
        "similaire, précise ce qui les distingue concrètement (le sujet abordé, "
        "l'intensité, le moment dans l'appel). Un rapport où chaque moment se lit "
        "comme une variation du précédent n'apporte rien.\n"
        "PRUDENCE SUR LE REGARD : ne jamais présenter un regard fuyant "
        "(gaze_aversion) comme un indice de mensonge ou de malhonnêteté — la "
        "méta-analyse de référence sur les indices de tromperie (DePaulo et al., "
        "2003, 'Cues to Deception') montre que le regard est l'un des signaux les "
        "MOINS fiables pour détecter un mensonge, malgré la croyance populaire "
        "répandue. Un regard qui se détourne se lit uniquement comme une charge "
        "cognitive, une réflexion, ou un malaise ponctuel — jamais comme une "
        "preuve de sincérité douteuse."
    ))
    confidence: str = Field(description="low, medium ou high")


class BehaviorReport(BaseModel):
    executive_summary: str = Field(description="résumé en 3-5 phrases pour un décideur pressé")
    overall_scores: Scores
    key_moments: list[KeyMoment] = Field(description=(
        "moments clés du call, triés chronologiquement, JUSQU'À 12 maximum. "
        "Ce n'est PAS un quota à remplir : n'inclus que les moments qui apportent "
        "une lecture réellement distincte. Sur un appel court ou globalement stable "
        "(peu de variation d'un moment à l'autre), il est normal et préférable "
        "d'en avoir seulement 2 ou 3 — mieux vaut peu de moments vraiment "
        "informatifs que remplir la liste en dupliquant la même observation avec "
        "des mots différents. Si le call ne contient pas assez de signal pour "
        "distinguer plus de moments, dis-le dans executive_summary (ex: 'état "
        "globalement stable tout au long de l'appel, sans variation notable') "
        "plutôt que d'inventer des nuances qui n'existent pas dans les données."
    ))
    closing_verdict: str = Field(default="", description=(
        "UNE phrase directe, sans détour : où en est CE deal — proche de "
        "signer, nécessite encore du travail, ou à risque réel — et QUELLE "
        "action unique est la plus déterminante MAINTENANT pour le faire "
        "avancer vers la signature. Ce n'est pas un résumé (executive_summary "
        "s'en charge déjà) : c'est un verdict opérationnel qui répond à 'qu'est-ce "
        "qu'on fait en priorité pour closer'. Base-toi sur buying_signal et "
        "deal_risk combinés aux key_moments, pas sur une impression générale. "
        "Exemple valide : 'Signal d'achat fort mais validation interne non "
        "actée — le blocage n'est plus commercial, obtenir explicitement le nom "
        "du décideur interne avant le prochain envoi.' Exemple invalide (trop "
        "vague) : 'Le deal avance bien, il faut rester à l'écoute.'"
    ))
    coaching_recommendations: list[str] = Field(description=(
        "2 à 5 actions CONCRÈTES pour faire avancer CE deal vers la "
        "signature — jamais des généralités de communication ('soyez à "
        "l'écoute', 'restez positif', 'instaurez la confiance') ni une "
        "évaluation de l'employé qui a mené l'appel. Chaque recommandation "
        "DOIT : (1) citer explicitement le key_moment ou le score précis de "
        "CE rapport qui la justifie — pas une intuition générale ; (2) donner "
        "une action ou une formulation VERBATIM à utiliser au prochain "
        "échange, jamais une intention vague. "
        "Mauvais exemple (vague, non actionnable) : 'Rassurez le client sur "
        "le prix.' "
        "Bon exemple (concret, actionnable, sourcé) : 'Le moment à 2:15 "
        "montre une hésitation sur le délai de livraison plutôt que sur le "
        "prix (signal d'achat qui reste haut juste après le chiffrage) — "
        "proposer un jalon intermédiaire daté avant de reparler du prix, qui "
        "n'est probablement pas le vrai obstacle ici.' "
        "Trie par priorité : la PREMIÈRE recommandation doit être l'action "
        "la plus déterminante pour faire avancer le closing, jamais la "
        "première trouvée dans l'ordre chronologique de l'appel."
    ))


class PreCallBrief(BaseModel):
    """Fiche de préparation pour le PROCHAIN appel avec ce même interlocuteur,
    générée à partir du rapport du dernier échange. Analyser après coup ne
    suffit pas — la vraie valeur est de bien se préparer avant, pas seulement
    de comprendre après."""

    context_recap: str = Field(description="rappel en 2-3 phrases de ce qui s'est passé au dernier échange, pour se remettre dans le contexte juste avant l'appel")
    likely_objections: list[str] = Field(description="2 à 5 objections probables au prochain appel, déduites des signaux d'hésitation ou de résistance observés au dernier échange — pas des objections génériques de vente")
    prep_actions: list[str] = Field(description="2 à 5 actions concrètes à préparer AVANT l'appel (un chiffrage, une réponse à une objection anticipée, un document à avoir sous la main) — jamais des généralités du type 'être à l'écoute'")
    opening_move: str = Field(description=(
        "une phrase concrète pour ouvrir le prochain appel en tenant compte de "
        "ce qui s'est passé au dernier échange — pas un template générique. "
        "Privilégie la technique du LABELING (Chris Voss, ex-négociateur du FBI, "
        "'Never Split the Difference', 2016) quand le rapport fourni contient un "
        "point de friction identifiable : nommer explicitement ce qui a semblé "
        "compter pour l'interlocuteur ('On dirait que la validation en interne "
        "était le vrai sujet la dernière fois, plus que le prix lui-même') plutôt "
        "que de le contourner ou d'attendre qu'il le reformule lui-même — nommer "
        "ce qu'il ressent ou ce qui le préoccupe désarme la défense et invite à "
        "confirmer ou corriger, sans jamais l'accuser ni le lui reprocher."
    ))
    watch_for: list[str] = Field(description="2-3 signaux précis à surveiller dès le début du prochain appel, parce qu'ils comptaient au dernier échange (ex: le moment où on reparle du budget)")
    verification_questions: list[str] = Field(default_factory=list, description=(
        "1 à 3 questions concrètes et directes à poser au prochain appel pour "
        "lever un doute resté NON tranché au dernier échange — cible en priorité "
        "les key_moments à confidence low ou medium du rapport fourni, où un "
        "signal non-verbal était ambigu sans confirmation verbale possible sur le "
        "moment. Principe directeur (méthode d'entretien d'investigation "
        "d'Ekman, reprise dans les formations FBI/BAI) : un signal non-verbal "
        "ambigu doit générer une HYPOTHÈSE À VÉRIFIER À L'ORAL, jamais une "
        "conclusion tranchée à sa place. Exemple valide : le rapport notait une "
        "hésitation (confidence medium) au moment du délai de livraison sans "
        "certitude sur la cause → 'Revenir explicitement sur le délai évoqué : "
        "poser la question ouverte permet de savoir si l'hésitation venait du "
        "délai lui-même ou d'un arbitrage interne non partagé.' Ne pas inventer "
        "de question si le rapport ne contient aucun moment réellement ambigu — "
        "une liste vide est préférable à des questions génériques de vente."
    ))
    previous_deal_risk: int = Field(default=0, description=(
        "NE PAS CALCULER CE CHAMP — il est écrasé programmatiquement après "
        "génération avec la valeur exacte de overall_scores.deal_risk du rapport "
        "fourni, pour garantir l'exactitude du chiffre affiché plutôt que de "
        "dépendre de la mémoire du modèle. Mets n'importe quelle valeur, elle "
        "sera ignorée."
    ))
    previous_buying_signal: int = Field(default=0, description=(
        "NE PAS CALCULER CE CHAMP — il est écrasé programmatiquement après "
        "génération avec la valeur exacte de overall_scores.buying_signal du "
        "rapport fourni, pour garantir l'exactitude du chiffre affiché plutôt "
        "que de dépendre de la mémoire du modèle. Mets n'importe quelle valeur, "
        "elle sera ignorée."
    ))
