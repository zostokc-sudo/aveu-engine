"""Image et environnement de la vidéo du CLOSER : uniquement ce qui se corrige en cinq minutes.

On mesure la lumière, le cadrage, la netteté, le fond et le centrage. Ce module NE JUGE JAMAIS le corps, le visage,
l'âge, l'origine ni les vêtements (caractéristiques protégées : discrimination, RGPD art. 9). Il n'est exécuté que si le
closer l'a demandé pour sa propre vidéo."""

import statistics
from pathlib import Path

import cv2
import numpy as np

from visual_analysis import MODELS_DIR


def _face_boxes(video_path: Path, duration_sec: float, samples: int):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
                                    running_mode=RunningMode.IMAGE, num_faces=1)
    cap = cv2.VideoCapture(str(video_path))
    out = []
    try:
        with FaceLandmarker.create_from_options(options) as lm:
            for i in range(samples):
                cap.set(cv2.CAP_PROP_POS_MSEC, duration_sec * 1000 * (i + 0.5) / samples)
                ok, frame = cap.read()
                if not ok:
                    continue
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if res.face_landmarks:
                    pts = res.face_landmarks[0]
                    xs, ys = [p.x for p in pts], [p.y for p in pts]
                    out.append((frame, (min(xs), min(ys), max(xs), max(ys))))
    finally:
        cap.release()
    return out


def frame_metrics(frame: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(max(0, box[0]) * w), int(max(0, box[1]) * h), int(min(1, box[2]) * w), int(min(1, box[3]) * h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = gray[y0:y1, x0:x1]
    mask = np.ones_like(gray, dtype=bool)
    px, py = int((x1 - x0) * 0.5), int((y1 - y0) * 0.7)  # le fond = tout sauf le visage élargi (tête, épaules)
    mask[max(0, y0 - py):min(h, y1 + py * 2), max(0, x0 - px):min(w, x1 + px)] = False
    edges = cv2.Canny(gray, 80, 160)
    bg_pixels = int(mask.sum()) or 1
    return {
        "face_h": (y1 - y0) / h,
        "center_x": (x0 + x1) / 2 / w,
        "center_y": (y0 + y1) / 2 / h,
        "face_light": float(face.mean()) if face.size else 0.0,
        "bg_light": float(gray[mask].mean()) if bg_pixels > 1 else 0.0,
        "sharp": float(cv2.Laplacian(cv2.resize(face, (128, 128)), cv2.CV_64F).var()) if face.size else 0.0,
        "clutter": float((edges[mask] > 0).sum()) / bg_pixels,
    }


def build_checks(m: dict) -> list[dict]:
    """Cinq vérifications, chacune avec un conseil simple. Aucun score global, aucun jugement sur la personne."""
    def item(label, ok, good, fix):
        return {"label": label, "ok": ok, "text": good if ok else fix}

    backlit = m["bg_light"] - m["face_light"] > 45
    return [
        item("Lumière sur le visage", 70 <= m["face_light"] <= 205 and not backlit,
             "Visage bien éclairé.",
             "Contre-jour : une fenêtre ou une lampe derrière vous assombrit le visage. Placez la lumière face à vous." if backlit
             else "Visage trop sombre ou trop clair. Une lampe douce face à vous règle presque tout."),
        item("Cadrage", 0.22 <= m["face_h"] <= 0.55,
             "Cadrage à bonne distance.",
             "Vous êtes trop loin : rapprochez-vous ou zoomez, le visage doit remplir environ un tiers de l'image." if m["face_h"] < 0.22
             else "Vous êtes trop près : reculez un peu, laissez de l'air au-dessus de la tête et sur les épaules."),
        item("Hauteur de la caméra", 0.28 <= m["center_y"] <= 0.55 and 0.35 <= m["center_x"] <= 0.65,
             "Caméra à hauteur des yeux, visage centré.",
             "Placez la caméra à hauteur des yeux (surélevez l'ordinateur) et centrez-vous dans l'image."),
        item("Netteté", m["sharp"] >= 25,
             "Image nette.", "Image un peu floue : nettoyez l'objectif, ajoutez de la lumière, vérifiez la qualité de la webcam."),
        item("Fond", m["clutter"] <= 0.09,
             "Fond sobre.", "Fond chargé : dégagez ce qui est derrière vous (ou utilisez un mur uni / un flou d'arrière-plan) pour que l'attention reste sur vous."),
    ]


def analyze_image(video_path: Path, duration_sec: float, samples: int = 10) -> dict | None:
    found = _face_boxes(video_path, duration_sec, samples)
    if len(found) < 3:
        return None
    ms = [frame_metrics(f, b) for f, b in found]
    med = {k: statistics.median(m[k] for m in ms) for k in ms[0]}
    checks = build_checks(med)
    return {"checks": checks, "ok_count": sum(c["ok"] for c in checks), "total": len(checks)}
