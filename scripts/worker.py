"""Traite les audits en attente (table `audits` de Supabase) : télécharge la vidéo du client,
lance le pipeline, dépose le rapport dans le stockage, supprime la vidéo.
Lancé toutes les 5 min par GitHub Actions (voir .github/workflows/audits.yml).
Variables : SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY."""

import contextlib
import io
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import MultipleFacesDetected, NoFaceDetected, NoUsableContent, run  # noqa: E402

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


def patch(audit_id: str, **fields) -> None:
    r = requests.patch(f"{URL}/rest/v1/audits?id=eq.{audit_id}", headers={**H, "Content-Type": "application/json"}, json=fields, timeout=30)
    r.raise_for_status()


def fetch_benchmark() -> dict | None:
    """Moyennes des appels déjà analysés (anonymes), disponibles à partir de 10 appels."""
    try:
        r = requests.post(f"{URL}/rest/v1/rpc/debrief_benchmark", headers={**H, "Content-Type": "application/json"}, json={}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print("benchmark indisponible :", repr(exc), file=sys.stderr)
        return None


def save_stats(audit_id: str, report_path: Path) -> None:
    stats_file = Path(report_path).parent / "stats.json"
    if not stats_file.exists():
        return
    import json
    row = {"audit_id": audit_id, **json.loads(stats_file.read_text(encoding="utf-8"))}
    requests.post(f"{URL}/rest/v1/debrief_stats", headers={**H, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}, json=row, timeout=30)


def part_paths(a: dict) -> list[str]:
    n = int(a.get("parts") or 1)
    return [a["video_path"]] if n == 1 else [f"{a['video_path']}.p{i:02d}" for i in range(n)]


class Tee(io.TextIOBase):
    """Copie tout ce que le pipeline affiche : garde un journal par audit et publie l'étape en cours (champ `progress`)."""

    def __init__(self, audit_id: str, real):
        self.audit_id, self.real, self.parts, self.last = audit_id, real, [], ""

    def write(self, text: str) -> int:
        self.real.write(text)
        self.parts.append(text)
        for line in text.splitlines():
            line = line.strip()
            if line and (re.match(r"\[\d/7\]", line) or line.startswith(("Débrief", "Préparation", "Voix"))) and line != self.last:
                self.last = line
                try:
                    patch(self.audit_id, progress=line[:110])
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        self.real.flush()

    def text(self) -> str:
        return "".join(self.parts)[-20000:]


def save_log(audit_id: str, log: str) -> None:
    try:
        requests.post(f"{URL}/rest/v1/audit_logs", headers={**H, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
                      json={"audit_id": audit_id, "log": log or "(vide)"}, timeout=30)
    except Exception as exc:
        print("journal non enregistré :", repr(exc), file=sys.stderr)


def process(a: dict) -> None:
    audit_id = a["id"]
    patch(audit_id, status="running", progress="Démarrage…")
    tee = Tee(audit_id, sys.stdout)
    tmp = Path(tempfile.mkdtemp())
    video = tmp / f"{audit_id}{Path(a['video_path']).suffix or '.mp4'}"
    try:
        with open(video, "wb") as f:  # vidéo envoyée en morceaux de 45 Mo (limite du stockage gratuit) : on recolle
            for part in part_paths(a):
                with requests.get(f"{URL}/storage/v1/object/videos/{part}", headers=H, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        with contextlib.redirect_stdout(tee):
          report = run(video, a["subject"], a["context"] or "appel de vente (interlocuteur externe)", a.get("source_url"),
                       filmed_role=a.get("filmed_role") or "prospect", focus=tuple(a.get("focus") or ()),
                       analyze_my_image=bool(a.get("analyze_image")), benchmark=fetch_benchmark())
        report_path = f"{a['user_id']}/{audit_id}.html"
        try:
            save_stats(audit_id, report)
        except Exception as exc:  # les statistiques ne doivent jamais empêcher la livraison du rapport
            print("stats non enregistrées :", repr(exc), file=sys.stderr)
        up = requests.post(
            f"{URL}/storage/v1/object/reports/{report_path}",
            headers={**H, "Content-Type": "text/html", "x-upsert": "true"}, data=Path(report).read_bytes(), timeout=120,
        )
        up.raise_for_status()
        patch(audit_id, status="done", report_path=report_path, progress=None)
    except (MultipleFacesDetected, NoFaceDetected, NoUsableContent) as exc:
        patch(audit_id, status="error", error=str(exc)[:300], progress=None)
    except Exception as exc:  # message court côté client, détail dans les logs Actions
        print("ERREUR", audit_id, repr(exc), file=sys.stderr)
        traceback.print_exc()
        patch(audit_id, status="error", error="L'analyse a échoué. Réessayez avec une autre vidéo.", progress=None)
    finally:
        save_log(audit_id, tee.text() + ("\n" + traceback.format_exc() if sys.exc_info()[0] else ""))
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
    for n, a in enumerate(r.json()):
        if n:
            time.sleep(60)  # plafond gratuit Groq par minute : on laisse la fenêtre se vider entre deux audits
        process(a)


if __name__ == "__main__":
    main()
