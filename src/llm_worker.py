"""Exécute la génération du rapport dans un processus séparé.

Utilise Groq (gratuit, cloud, Kimi K2) si GROQ_API_KEY est configurée —
c'est le chemin qualité. Sinon, retombe sur le modèle local (Qwen2.5-7B,
qualité de démonstration). Voir vision_worker.py pour l'explication de
l'isolation en processus séparé : elle ne sert qu'au chemin local (mediapipe
garde des buffers Metal en mémoire, ce qui affamait un modèle local un peu
gros s'il tournait dans le même processus) — un appel Groq est une simple
requête HTTP, mais on garde la même isolation pour ne pas bifurquer
l'architecture selon le chemin emprunté.

Usage : python3 llm_worker.py <timeline.json> <out_report.json> [--model ID] [--context TEXTE]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from groq_llm import parse as groq_parse
from local_llm import GenerationTruncated, MODEL_ID, generate_report_local
from quality_check import quality_warnings
from report_generator import SYSTEM_PROMPT, _compact_for_prompt, select_salient_windows
from schema import BehaviorReport


def _generate_via_groq(timeline: list[dict], context: str) -> BehaviorReport:
    salient = select_salient_windows(timeline)
    return groq_parse(
        system=SYSTEM_PROMPT.format(context=context),
        user=(
            "Voici la timeline complète de l'enregistrement, en JSON. "
            "Produis le rapport d'analyse comportementale.\n\n"
            + json.dumps(_compact_for_prompt(salient), ensure_ascii=False)
        ),
        output_format=BehaviorReport,
        max_tokens=4000,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline_json", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--context", default="entretien de vente B2B (interlocuteur externe)")
    args = parser.parse_args()

    timeline = json.loads(args.timeline_json.read_text(encoding="utf-8"))
    report = None

    if os.environ.get("GROQ_API_KEY"):
        try:
            report = _generate_via_groq(timeline, args.context)
        except Exception as e:
            print(f"GROQ INDISPONIBLE ({e}) — bascule sur le modèle local.", file=sys.stderr)

    if report is None:
        try:
            report = generate_report_local(timeline, model_id=args.model)
        except GenerationTruncated as e:
            print(f"TRUNCATED: {e}", file=sys.stderr)
            return 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(report.model_dump_json(), encoding="utf-8")
    for warning in quality_warnings(report, timeline):
        print(f"QUALITY WARNING: {warning}", file=sys.stderr)
    print(f"OK -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
