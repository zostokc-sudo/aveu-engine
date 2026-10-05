"""API d'audit hébergeable (Linux, CPU, gratuit) : reçoit la vidéo d'un client connecté
via Supabase, lance le pipeline en arrière-plan, sert le rapport HTML.

Variables d'environnement : SUPABASE_URL, SUPABASE_ANON_KEY, GROQ_API_KEY, ALLOWED_ORIGIN.
"""

import os
import threading
import uuid
from pathlib import Path

import requests
from flask import Flask, abort, jsonify, request, send_file
from werkzeug.utils import secure_filename

from pipeline import WORK_DIR, run

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3
UPLOADS = WORK_DIR / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict] = {}  # job_id -> {"user": id, "status": ..., "report": Path|None, "error": str|None}


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("ALLOWED_ORIGIN", "https://aveu.surge.sh")
    resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    return resp


def current_user() -> str:
    """Vérifie le jeton Supabase du client ; renvoie son id ou coupe la requête."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        abort(401)
    r = requests.get(
        os.environ["SUPABASE_URL"] + "/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": os.environ["SUPABASE_ANON_KEY"]},
        timeout=10,
    )
    if r.status_code != 200:
        abort(401)
    return r.json()["id"]


def _work(job_id: str, video: Path, subject: str, context: str) -> None:
    try:
        JOBS[job_id]["report"] = run(video, subject, context)
        JOBS[job_id]["status"] = "done"
    except Exception as exc:  # affiché au client, jamais de stacktrace brute
        JOBS[job_id].update(status="error", error=str(exc)[:300])
    finally:
        video.unlink(missing_ok=True)  # la vidéo du client n'est pas conservée


@app.post("/audit")
def audit():
    user = current_user()
    f = request.files.get("video")
    if not f:
        abort(400)
    job_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{job_id}-{secure_filename(f.filename or 'appel.mp4')}"
    f.save(path)
    JOBS[job_id] = {"user": user, "status": "running", "report": None, "error": None}
    threading.Thread(
        target=_work,
        args=(job_id, path, request.form.get("subject", "Interlocuteur analysé"), request.form.get("context", "")),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id), 202


@app.get("/audit/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job or job["user"] != current_user():
        abort(404)
    return jsonify(status=job["status"], error=job["error"])


@app.get("/audit/<job_id>/report")
def report(job_id):
    job = JOBS.get(job_id)
    if not job or job["user"] != current_user() or job["status"] != "done":
        abort(404)
    return send_file(job["report"])


@app.route("/audit", methods=["OPTIONS"])
@app.route("/audit/<path:_>", methods=["OPTIONS"])
def preflight(_=None):
    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
