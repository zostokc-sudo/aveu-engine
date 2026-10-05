"""Génère une fiche de préparation pour le prochain appel, à partir du
rapport du dernier échange — dans un processus séparé (voir vision_worker.py
pour l'explication de l'isolation mémoire, qui ne concerne que le chemin
local ; voir llm_worker.py pour la même bascule Groq/local).

Usage : python3 precall_worker.py <previous_report.json> <out_brief.json> [--context "..."] [--model ID]
"""

import argparse
import os
import sys
from pathlib import Path

from groq_llm import parse as groq_parse
from local_llm import MODEL_ID, GenerationTruncated, generate_precall_brief_local
from quality_check import precall_quality_warnings
from report_generator import PRECALL_SYSTEM_PROMPT
from schema import BehaviorReport, PreCallBrief


def _generate_via_groq(previous_report: BehaviorReport, context: str) -> PreCallBrief:
    brief = groq_parse(
        system=PRECALL_SYSTEM_PROMPT,
        user=(
            f"Contexte du prochain appel : {context}\n\n"
            "Voici le rapport du dernier échange avec cet interlocuteur, en JSON :\n\n"
            + previous_report.model_dump_json()
        ),
        output_format=PreCallBrief,
        max_tokens=2000,
    )
    brief.previous_deal_risk = previous_report.overall_scores.deal_risk
    brief.previous_buying_signal = previous_report.overall_scores.buying_signal
    return brief


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("previous_report_json", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--context", default="prochain appel de vente")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    previous_report = BehaviorReport.model_validate_json(args.previous_report_json.read_text(encoding="utf-8"))
    brief = None

    if os.environ.get("GROQ_API_KEY"):
        try:
            brief = _generate_via_groq(previous_report, args.context)
        except Exception as e:
            print(f"GROQ INDISPONIBLE ({e}) — bascule sur le modèle local.", file=sys.stderr)

    if brief is None:
        try:
            brief = generate_precall_brief_local(previous_report, context=args.context, model_id=args.model)
        except GenerationTruncated as e:
            print(f"TRUNCATED: {e}", file=sys.stderr)
            return 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(brief.model_dump_json(), encoding="utf-8")
    for warning in precall_quality_warnings(brief):
        print(f"QUALITY WARNING: {warning}", file=sys.stderr)
    print(f"OK -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
