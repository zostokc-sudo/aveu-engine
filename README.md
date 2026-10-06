# aveu-engine

Moteur d'analyse d'[Aveu](https://aveu.surge.sh), exécuté gratuitement par GitHub Actions.

**Fonctionnement.** Un client envoie sa vidéo depuis le site (stockage Supabase, morceaux de 45 Mo) et crée une ligne dans `audits`
(statut `queued`). Toutes les 5 minutes, `.github/workflows/audits.yml` vérifie s'il y a du travail ; si oui, `scripts/worker.py`
recolle la vidéo, lance `src/pipeline.py` (transcription faster-whisper, séparation des voix sherpa-onnx, visage/posture MediaPipe,
rapport via Groq), dépose le rapport HTML dans le bucket `reports` et supprime la vidéo.

**Secrets Actions requis :** `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY`.

**À savoir.**
- GitHub désactive les workflows planifiés d'un dépôt public après 60 jours sans activité : pousser un commit tous les ~2 mois.
- Plafond gratuit Groq : 8000 tokens/minute ; `generate_report` réduit la requête tout seul.
- Quota : 5 audits par compte et par 24 h (politique RLS Supabase `audits_quota_ok`).
- Les analyses ne portent que sur l'interlocuteur externe (AI Act, art. 5).
