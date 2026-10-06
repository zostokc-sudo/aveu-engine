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


def part_paths(a: dict) -> list[str]:
    n = int(a.get("parts") or 1)
    return [a["video_path"]] if n == 1 else [f"{a['video_path']}.p{i:02d}" for i in range(n)]


def process(a: dict) -> None:
    audit_id = a["id"]
    patch(audit_id, status="running")
    tmp = Path(tempfile.mkdtemp())
    video = tmp / f"{audit_id}{Path(a['video_path']).suffix or '.mp4'}"
    try:
        with open(video, "wb") as f:  # vidéo envoyée en morceaux de 45 Mo (limite du stockage gratuit) : on recolle
            for part in part_paths(a):
                with requests.get(f"{URL}/storage/v1/object/videos/{part}", headers=H, stream=True, timeout=300) as r:
                    r.raise_for_status()
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
        for part in part_paths(a):  # on ne garde pas la vidéo
            requests.delete(f"{URL}/storage/v1/object/videos/{part}", headers=H, timeout=30)


def release_stuck() -> None:
    """Un audit bloqué en 'running' (job GitHub interrompu) repasse en erreur au bout de 90 min."""
    from datetime import datetime, timedelta, timezone
    limit = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    r = requests.patch(
        f"{URL}/rest/v1/audits",
        params={"status": "eq.running", "created_at": f"lt.{limit}"},  # params : le '+' du fuseau est encodé correctement
        headers={**H, "Content-Type": "application/json"},
        json={"status": "error", "error": "L'analyse a été interrompue. Renvoyez votre vidéo."}, timeout=30,
    )
    r.raise_for_status()


def main() -> None:
    try:
        release_stuck()
    except Exception as exc:  # jamais bloquer le traitement pour un nettoyage
        print("release_stuck a échoué :", repr(exc), file=sys.stderr)
    r = requests.get(f"{URL}/rest/v1/audits?status=eq.queued&order=created_at.asc&limit=3", headers=H, timeout=30)
    r.raise_for_status()
    for a in r.json():
        process(a)


if __name__ == "__main__":
    main()
