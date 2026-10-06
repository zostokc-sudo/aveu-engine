"""Orchestrateur : vidéo -> transcription + signaux comportementaux -> rapport HTML."""

import argparse
from pathlib import Path

from audio_analysis import analyze_audio
from diarize import diarize, filmed_speaker, label_segments
from extract_audio import extract_audio, probe_duration
from html_report import render_html, render_pdf
from report_generator import compute_behavior_profile, generate_report, save_report_json
from transcribe import transcribe
from visual_analysis import analyze_video, bucketize, check_multi_face_warning

WORK_DIR = Path(__file__).parent.parent / "output"


class MultipleFacesDetected(RuntimeError):
    """La vidéo montre plusieurs personnes à l'écran une part significative du
    temps — on refuse de générer un rapport tant que ce n'est pas corrigé,
    pour ne jamais risquer d'analyser le mauvais interlocuteur."""


def run(video_path: Path, subject_name: str, context: str) -> Path:
    stem = video_path.stem
    run_dir = WORK_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Extraction audio…")
    wav_path = extract_audio(video_path, run_dir / "audio.wav")
    duration = probe_duration(video_path)

    print(f"[2/7] Transcription…")
    segments = transcribe(wav_path)

    print(f"[3/7] Analyse vocale (hauteur, énergie)…")
    audio_buckets = analyze_audio(wav_path)

    print(f"[4/7] Analyse visuelle (visage + posture)…")
    frames = analyze_video(video_path)
    buckets = bucketize(frames)

    warning = check_multi_face_warning(buckets)
    if warning:
        raise MultipleFacesDetected(warning)

    turns = diarize(wav_path)
    if label_segments(segments, turns, filmed_speaker(turns, buckets)):
        print("Voix séparées : la personne filmée est identifiée.")
    else:
        print("Voix non séparées (une seule voix, modèles absents ou doute) : texte non étiqueté.")

    print(f"[5/7] Génération du rapport (Claude)…")
    report = generate_report(segments, buckets, context=context, audio_buckets=audio_buckets)
    save_report_json(report, run_dir / "report.json")

    print(f"[6/7] Rendu HTML…")
    html_path = render_html(
        report, run_dir / "report.html",
        subject_name=subject_name, context=context, duration_min=duration / 60,
        behavior_profile=compute_behavior_profile(buckets),
    )

    print(f"[7/7] Export PDF…")
    pdf_path = render_pdf(html_path)

    print(f"Terminé : {html_path} / {pdf_path}")
    return html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse comportementale d'une vidéo")
    parser.add_argument("video", type=Path, help="chemin de la vidéo (doit montrer le prospect/interlocuteur externe, jamais un salarié du client)")
    parser.add_argument("--subject", default="Interlocuteur analysé", help="nom du prospect/interlocuteur pour le rapport")
    parser.add_argument("--context", default="entretien de vente B2B (interlocuteur externe)", help="contexte de l'enregistrement")
    args = parser.parse_args()
    run(args.video, args.subject, args.context)
