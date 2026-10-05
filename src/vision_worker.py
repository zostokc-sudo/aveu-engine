"""Exécute l'analyse vidéo (mediapipe) dans un processus séparé et jetable.

Pourquoi : mediapipe garde des buffers Metal/GPU en mémoire pendant toute la
vie du process Python, même après la fermeture de ses context managers. Sur
une machine à 8 Go de RAM unifiée, ça laisse trop peu de marge pour charger
ensuite un modèle de langage correct dans le MÊME processus (constaté : un
modèle 7B tient seul, mais plante en "Insufficient Memory" juste après
mediapipe). En lançant l'analyse vidéo dans un processus qui se termine
complètement une fois fini, l'OS récupère toute sa mémoire avant que le
modèle de langage ne démarre dans un processus neuf — ce qui permet d'utiliser
un modèle sensiblement plus gros et plus précis, gratuitement, sur la même
machine.

Usage : python3 vision_worker.py <video.mp4> <out_buckets.json> [--warning-only]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from visual_analysis import analyze_video, bucketize, check_multi_face_warning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("out_json", type=Path)
    args = parser.parse_args()

    frames = analyze_video(args.video)
    buckets = bucketize(frames)

    result = {
        "buckets": [dataclasses.asdict(b) for b in buckets],
        "multi_face_warning": check_multi_face_warning(buckets),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result), encoding="utf-8")
    print(f"OK: {len(buckets)} fenêtres -> {args.out_json}")


if __name__ == "__main__":
    sys.exit(main())
