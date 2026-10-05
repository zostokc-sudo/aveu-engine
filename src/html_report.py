"""Rendu du rapport structuré en HTML premium (livrable client)."""

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from radar_chart import RadarAxis, build_radar, score_color
from schema import BehaviorReport, PreCallBrief

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _build_radar_data(report: BehaviorReport, behavior_profile: dict[str, float]) -> dict:
    scores = report.overall_scores
    axes = [
        RadarAxis("engagement", "Engagement", scores.engagement, higher_is_better=True),
        RadarAxis("openness", "Ouverture", scores.openness, higher_is_better=True),
        RadarAxis("buying_signal", "Signal d'achat", scores.buying_signal, higher_is_better=True),
        RadarAxis("deal_risk", "Risque affaire", scores.deal_risk, higher_is_better=False),
        RadarAxis("gaze_stability", "Regard stable", behavior_profile.get("gaze_stability", 0.0), higher_is_better=True),
        RadarAxis("genuine_smile", "Sourire authentique", behavior_profile.get("genuine_smile", 0.0), higher_is_better=True),
        RadarAxis("gesture_stability", "Stabilité gestuelle", behavior_profile.get("gesture_stability", 0.0), higher_is_better=True),
    ]
    return build_radar(axes)


def render_html(
    report: BehaviorReport,
    out_path: Path,
    subject_name: str,
    context: str,
    duration_min: float,
    behavior_profile: dict[str, float] | None = None,
) -> Path:
    behavior_profile = behavior_profile or {}
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))
    env.filters["score_color"] = score_color
    template = env.get_template("report_template.html")
    html = template.render(
        report=report.model_dump(),
        subject_name=subject_name,
        context=context,
        duration_min=round(duration_min, 1),
        generated_date=date.today().isoformat(),
        radar=_build_radar_data(report, behavior_profile),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_pdf(html_path: Path, out_pdf_path: Path | None = None) -> Path:
    """Convertit un rapport HTML déjà rendu en PDF (weasyprint, gratuit/local).
    Un acheteur à ce niveau de prix attend un vrai livrable, pas juste une page
    web — import de weasyprint différé ici pour ne pas alourdir le chargement
    de ce module partout ailleurs où il n'est pas nécessaire."""
    import weasyprint

    out_pdf_path = out_pdf_path or html_path.with_suffix(".pdf")
    weasyprint.HTML(filename=str(html_path)).write_pdf(str(out_pdf_path))
    return out_pdf_path


def render_precall_html(
    brief: PreCallBrief,
    out_path: Path,
    subject_name: str,
    context: str = "prochain appel de vente",
) -> Path:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("precall_template.html")
    html = template.render(
        brief=brief.model_dump(),
        subject_name=subject_name,
        context=context,
        generated_date=date.today().isoformat(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
