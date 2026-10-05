"""Traite les audits en attente (table `audits` de Supabase) : télécharge la vidéo du client,
lance le pipeline, dépose le rapport dans le stockage, supprime la vidéo.
Lancé toutes les 5 min par GitHub Actions (voir .github/workflows/audits.yml).
Variables : SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY."""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import MultipleFacesDetected, run  # noqa: E402

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


def patch(audit_id: str, **fields) -> None:
    r = requests.patch(f"{URL}/rest/v1/audits?id=eq.{audit_id}", headers={**H, "Content-Type": "application/json"}, json=fields, timeout=30)
    r.raise_for_status()


def process(a: dict) -> None:
    audit_id = a["id"]
    patch(audit_id, status="running")
    tmp = Path(tempfile.mkdtemp())
    video = tmp / f"{audit_id}{Path(a['video_path']).suffix or '.mp4'}"
    try:
        with requests.get(f"{URL}/storage/v1/object/videos/{a['video_path']}", headers=H, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(video, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        report = run(video, a["subject"], a["context"] or "appel de vente (interlocuteur externe)")
        report_path = f"{a['user_id']}/{audit_id}.html"
        up = requests.post(
            f"{URL}/storage/v1/object/reports/{report_path}",
            headers={**H, "Content-Type": "text/html", "x-upsert": "true"}, data=Path(report).read_bytes(), timeout=120,
        )
        up.raise_for_status()
        patch(audit_id, status="done", report_path=report_path)
    except MultipleFacesDetected as exc:
        patch(audit_id, status="error", error=str(exc)[:300])
    except Exception as exc:  # message court côté client, détail dans les logs Actions
        print("ERREUR", audit_id, repr(exc), file=sys.stderr)
        traceback.print_exc()
        patch(audit_id, status="error", error="L'analyse a échoué. Réessayez avec une autre vidéo.")
    finally:
        requests.delete(f"{URL}/storage/v1/object/videos/{a['video_path']}", headers=H, timeout=30)  # on ne garde pas la vidéo


def main() -> None:
    r = requests.get(f"{URL}/rest/v1/audits?status=eq.queued&order=created_at.asc&limit=3", headers=H, timeout=30)
    r.raise_for_status()
    for a in r.json():
        process(a)


if __name__ == "__main__":
    main()
