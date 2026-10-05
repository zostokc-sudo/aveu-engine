"""CLI : génère la fiche de préparation pour le prochain appel, à partir du
rapport JSON du dernier échange (produit par pipeline.py --> report.json).

Usage : python3 precall_cli.py output/mon_appel/report.json --subject "..."
"""

import argparse
from pathlib import Path

from html_report import render_pdf, render_precall_html
from report_generator import generate_precall_brief
from schema import BehaviorReport


def main():
    parser = argparse.ArgumentParser(description="Génère une fiche de préparation pour le prochain appel")
    parser.add_argument("report_json", type=Path, help="rapport du dernier échange (report.json de pipeline.py)")
    parser.add_argument("--subject", default="Interlocuteur analysé")
    parser.add_argument("--context", default="prochain appel de vente")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    previous_report = BehaviorReport.model_validate_json(args.report_json.read_text(encoding="utf-8"))
    brief = generate_precall_brief(previous_report, context=args.context)

    out_path = args.out or args.report_json.parent / "precall.html"
    html_path = render_precall_html(brief, out_path, subject_name=args.subject, context=args.context)
    pdf_path = render_pdf(html_path)
    print(f"Terminé : {html_path} / {pdf_path}")


if __name__ == "__main__":
    main()
