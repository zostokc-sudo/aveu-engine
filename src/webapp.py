"""Interface web locale minimale pour tester le pipeline Aveu sur ses propres
vidéos, sans ligne de commande. Utilise Groq (gratuit, cloud, Kimi K2) si
GROQ_API_KEY est configurée — sinon retombe sur un modèle local (Qwen2.5-7B,
gratuit aussi, mais qualité de démonstration).

Vision et génération tournent chacune dans un processus séparé et jetable
(voir vision_worker.py / llm_worker.py) : mediapipe garde des buffers Metal
en mémoire pendant toute la vie d'un processus, même après la fermeture de
ses context managers. Sur une machine à 8 Go de RAM unifiée, ça faisait
planter un modèle local un peu gros s'il tournait dans le même processus.
En isolant chaque étape lourde, l'OS récupère toute la mémoire entre les
deux — ce qui permet d'utiliser un modèle plus gros, gratuitement. Cette
isolation ne sert qu'au chemin local ; un appel Groq est une simple requête
HTTP, mais on garde la même architecture pour ne pas bifurquer selon le
chemin emprunté.
"""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template_string, request, send_file
from werkzeug.utils import secure_filename

import groq_llm  # noqa: F401 (le seul effet recherché ici est de charger .env au démarrage)
from audio_analysis import analyze_audio
from extract_audio import extract_audio, probe_duration
from html_report import render_html, render_pdf, render_precall_html
from local_llm import MODEL_ID
from report_generator import compute_behavior_profile, merge_timeline
from schema import BehaviorReport, PreCallBrief
from transcribe import transcribe
from visual_analysis import buckets_from_dicts

SRC_DIR = Path(__file__).parent
PYTHON = sys.executable
WORK_DIR = Path(__file__).parent.parent / "output" / "webapp"
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 Go

# Machine à 8 Go de RAM unifiée : deux analyses en parallèle (double-clic,
# double soumission du formulaire) épuisent la mémoire Metal et plantent.
# Un verrou garantit qu'une seule analyse tourne à la fois ; les autres
# reçoivent un message clair plutôt qu'un crash.
_analysis_lock = threading.Lock()

UPLOAD_PAGE = """
<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>Aveu — test local</title>
<style>
  body { background:#0a0f1a; color:#eef1f6; font-family:-apple-system,sans-serif; max-width:640px; margin:60px auto; padding:0 24px; }
  h1 { font-size:24px; } p { color:#a7b2c6; line-height:1.5; }
  .box { background:#121a2b; border:1px solid #263048; border-radius:6px; padding:28px; margin-top:24px; }
  input[type=file] { width:100%; padding:12px; background:#182337; border:1px solid #263048; border-radius:4px; color:#eef1f6; }
  input[type=text] { width:100%; padding:10px; background:#182337; border:1px solid #263048; border-radius:4px; color:#eef1f6; margin-top:6px; }
  label { font-size:13px; color:#a7b2c6; display:block; margin-top:16px; }
  button { margin-top:22px; background:#cdae6c; color:#17140a; border:none; padding:12px 24px; border-radius:4px; font-weight:600; cursor:pointer; }
  .warn { background:#2a2013; border:1px solid #8a7546; color:#cdae6c; padding:10px 14px; border-radius:4px; font-size:13px; margin-top:16px; }
</style></head>
<body>
  <h1>av<span style="color:#cdae6c">eu</span> — test local</h1>
  <p>Envoie une vidéo (idéalement 30s-2min, un visage bien visible face caméra) pour voir le pipeline complet tourner : transcription, analyse comportementale, rapport.</p>
  {mode_banner}
  <div class="box">
    <form method="post" action="/analyze" enctype="multipart/form-data">
      <label>Vidéo</label>
      <input type="file" name="video" accept="video/*" required>
      <label>Nom du sujet (optionnel)</label>
      <input type="text" name="subject" placeholder="ex: Prospect appel du 12/09">
      <label>Contexte (optionnel)</label>
      <input type="text" name="context" value="entretien de vente B2B (interlocuteur externe)">
      <button type="submit" id="submit-btn">Lancer l'analyse (peut prendre quelques minutes)</button>
    </form>
  </div>
  <script>
    document.querySelector("form").addEventListener("submit", function () {
      var btn = document.getElementById("submit-btn");
      btn.disabled = true;
      btn.textContent = "Analyse en cours… (ne pas recharger la page)";
    });
  </script>
</body></html>
"""


@app.get("/")
def index():
    if os.environ.get("GROQ_API_KEY"):
        banner = ""
    else:
        banner = (
            '<div class="warn">Mode local gratuit — qualité de démonstration. '
            "Configure GROQ_API_KEY pour la qualité de production, gratuite aussi "
            '(<a href="https://console.groq.com/keys" style="color:#cdae6c">console.groq.com/keys</a>).</div>'
        )
    return render_template_string(UPLOAD_PAGE.replace("{mode_banner}", banner))


@app.post("/analyze")
def analyze():
    file = request.files.get("video")
    if not file or not file.filename:
        return redirect("/")

    if not _analysis_lock.acquire(blocking=False):
        return (
            "Une analyse est déjà en cours sur cette machine (mémoire limitée, "
            "une seule à la fois). Réessaie dans une minute.",
            429,
        )

    try:
        run_id = uuid4().hex[:8]
        run_dir = WORK_DIR / run_id
        run_dir.mkdir(parents=True)
        safe_filename = secure_filename(file.filename) or "video.mp4"
        video_path = run_dir / safe_filename
        file.save(video_path)

        subject = request.form.get("subject") or "Interlocuteur analysé"
        context = request.form.get("context") or "entretien de vente B2B (interlocuteur externe)"

        wav_path = extract_audio(video_path, run_dir / "audio.wav")
        duration = probe_duration(video_path)
        segments = transcribe(wav_path)
        audio_buckets = analyze_audio(wav_path)

        buckets_json = run_dir / "buckets.json"
        vision = subprocess.run(
            [PYTHON, str(SRC_DIR / "vision_worker.py"), str(video_path), str(buckets_json)],
            cwd=SRC_DIR, capture_output=True, text=True,
        )
        if vision.returncode != 0:
            return f"Échec de l'analyse visuelle : {vision.stderr[-500:]}", 500

        vision_result = json.loads(buckets_json.read_text(encoding="utf-8"))
        buckets = buckets_from_dicts(vision_result["buckets"])

        if vision_result["multi_face_warning"]:
            return vision_result["multi_face_warning"], 400

        timeline = merge_timeline(segments, buckets, audio_buckets)
        if not timeline:
            return "Aucun visage détecté de façon fiable dans cette vidéo. Essaie avec un cadrage plus net, visage face caméra.", 400

        timeline_json = run_dir / "timeline.json"
        timeline_json.write_text(json.dumps(timeline), encoding="utf-8")
        report_json = run_dir / "report.json"
        llm = subprocess.run(
            [PYTHON, str(SRC_DIR / "llm_worker.py"), str(timeline_json), str(report_json),
             "--model", MODEL_ID, "--context", context],
            cwd=SRC_DIR, capture_output=True, text=True,
        )
        if llm.returncode != 0:
            return (
                f"La génération du rapport a échoué ({llm.stderr[-500:]}). Réessaie.",
                500,
            )

        report = BehaviorReport.model_validate_json(report_json.read_text(encoding="utf-8"))
        (run_dir / "subject.txt").write_text(subject, encoding="utf-8")

        html_path = render_html(
            report, run_dir / "report.html",
            subject_name=subject, context=context, duration_min=duration / 60,
            behavior_profile=compute_behavior_profile(buckets),
        )
        render_pdf(html_path, run_dir / "report.pdf")
        html = html_path.read_text(encoding="utf-8")
        action_bar = (
            f'<div style="max-width:900px;margin:0 auto;padding:0 56px 40px;display:flex;gap:12px;flex-wrap:wrap">'
            f'<a href="/precall/{run_id}" style="display:inline-block;background:#b8934f;'
            f'color:#0f1e35;font-weight:600;padding:12px 22px;border-radius:4px;'
            f'text-decoration:none;font-family:-apple-system,sans-serif">'
            f"Générer la fiche de préparation pour le prochain appel</a>"
            f'<a href="/download/{run_id}/report.pdf" style="display:inline-block;background:transparent;'
            f'border:1px solid #b8934f;color:#b8934f;font-weight:600;padding:11px 22px;border-radius:4px;'
            f'text-decoration:none;font-family:-apple-system,sans-serif">'
            f"Télécharger en PDF</a></div>"
        )
        html = html.replace("</body>", action_bar + "</body>")
        html_path.write_text(html, encoding="utf-8")
        return send_file(html_path)
    finally:
        _analysis_lock.release()


@app.get("/precall/<run_id>")
def precall(run_id: str):
    if not re.fullmatch(r"[0-9a-f]{8}", run_id):
        return "Identifiant invalide.", 400

    run_dir = WORK_DIR / run_id
    report_json = run_dir / "report.json"
    if not report_json.exists():
        return "Rapport introuvable pour cet identifiant.", 404

    if not _analysis_lock.acquire(blocking=False):
        return "Une analyse est déjà en cours sur cette machine. Réessaie dans une minute.", 429

    try:
        subject = (run_dir / "subject.txt").read_text(encoding="utf-8") if (run_dir / "subject.txt").exists() else "Interlocuteur analysé"
        brief_json = run_dir / "precall_brief.json"
        result = subprocess.run(
            [PYTHON, str(SRC_DIR / "precall_worker.py"), str(report_json), str(brief_json), "--model", MODEL_ID],
            cwd=SRC_DIR, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return f"Échec de la génération de la fiche : {result.stderr[-500:]}", 500

        brief = PreCallBrief.model_validate_json(brief_json.read_text(encoding="utf-8"))
        html_path = render_precall_html(brief, run_dir / "precall.html", subject_name=subject)
        render_pdf(html_path, run_dir / "precall.pdf")
        html = html_path.read_text(encoding="utf-8")
        pdf_link = (
            f'<div style="max-width:760px;margin:0 auto;padding:0 48px 40px">'
            f'<a href="/download/{run_id}/precall.pdf" style="display:inline-block;background:transparent;'
            f'border:1px solid #b8934f;color:#b8934f;font-weight:600;padding:11px 22px;border-radius:4px;'
            f'text-decoration:none;font-family:-apple-system,sans-serif">'
            f"Télécharger en PDF</a></div>"
        )
        html = html.replace("</body>", pdf_link + "</body>")
        html_path.write_text(html, encoding="utf-8")
        return send_file(html_path)
    finally:
        _analysis_lock.release()


@app.get("/download/<run_id>/<filename>")
def download(run_id: str, filename: str):
    if not re.fullmatch(r"[0-9a-f]{8}", run_id):
        return "Identifiant invalide.", 400
    if filename not in ("report.pdf", "precall.pdf"):
        return "Fichier invalide.", 400

    file_path = WORK_DIR / run_id / filename
    if not file_path.exists():
        return "Fichier introuvable.", 404
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    # 0.0.0.0 plutôt que 127.0.0.1 : accessible depuis le téléphone sur le
    # même wifi (via l'IP locale du Mac), pas seulement depuis le Mac lui-même.
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=False, processes=1)
