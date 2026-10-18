"""Orchestrateur : vidéo -> transcription + signaux comportementaux -> rapport HTML."""

import argparse
from pathlib import Path

from audio_analysis import analyze_audio
from debrief import build_debrief
from conversation import conversation_metrics, transcript_lines
from diarize import diarize, filmed_speaker, label_segments
from extract_audio import extract_audio, normalize_video, probe_duration
from html_report import render_debrief_only, render_html, render_pdf
from report_generator import compute_behavior_profile, generate_report, save_report_json
from transcribe import transcribe
from visual_analysis import analyze_video, bucketize, check_multi_face_warning, probe_face_coverage

WORK_DIR = Path(__file__).parent.parent / "output"


class NoUsableContent(RuntimeError):
    """Ni image exploitable ni texte suffisant."""


class NoFaceDetected(RuntimeError):
    """Aucun visage exploitable dans la vidéo : on le dit tout de suite, avant de calculer quoi que ce soit."""


class MultipleFacesDetected(RuntimeError):
    """La vidéo montre plusieurs personnes à l'écran une part significative du
    temps — on refuse de générer un rapport tant que ce n'est pas corrigé,
    pour ne jamais risquer d'analyser le mauvais interlocuteur."""


def _no_face_note(reason: str) -> str:
    return (f"{reason} L'analyse des réactions du prospect (visage, regard, gestes) n'a donc pas été faite ; "
            "ce rapport porte uniquement sur ce que vous avez dit et fait, d'après la transcription.")


def run(video_path: Path, subject_name: str, context: str) -> Path:
    stem = video_path.stem
    run_dir = WORK_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Extraction audio…")
    wav_path = extract_audio(video_path, run_dir / "audio.wav")
    duration = probe_duration(video_path)

    print("Préparation de la vidéo (H.264 480p)…")
    visual_ok, visual_note = True, ""
    try:
        norm_path = normalize_video(video_path, run_dir / "video_480p.mp4")
        if probe_face_coverage(norm_path, duration) < 0.2:
            visual_ok, visual_note = False, _no_face_note("Aucun visage exploitable dans cette vidéo (plan trop large, partage d'écran ou audio seul).")
    except Exception as exc:  # audio seul, vidéo illisible : le texte suffit pour le débrief
        visual_ok, visual_note = False, _no_face_note("La partie image de ce fichier n'a pas pu être lue.")
        print(f"Image illisible ({exc!r}) : débrief sur le texte seul.")

    print("[2/7] Transcription…")
    segments = transcribe(wav_path)

    report = behavior_profile = buckets = audio_buckets = None
    if visual_ok:
        print("[3/7] Analyse vocale (hauteur, énergie)…")
        audio_buckets = analyze_audio(wav_path)

        print("[4/7] Analyse visuelle (visage + posture)…")
        buckets = bucketize(analyze_video(norm_path))
        warning = check_multi_face_warning(buckets)
        if warning:
            visual_ok, visual_note = False, _no_face_note("Plusieurs visages apparaissent à l'écran en même temps : impossible de suivre la même personne sans risque de confusion.")

    turns = diarize(wav_path)
    roles_known = label_segments(segments, turns, filmed_speaker(turns, buckets) if visual_ok else None)
    print("Voix séparées : la personne filmée est identifiée." if roles_known else "Voix non étiquetées : les rôles seront déduits du texte.")

    if visual_ok:
        print("[5/7] Rapport sur les réactions du prospect…")
        report = generate_report(segments, buckets, context=context, audio_buckets=audio_buckets)
        save_report_json(report, run_dir / "report.json")
        behavior_profile = compute_behavior_profile(buckets)

    print("[6/7] Débrief du closer (texte)…")
    debrief = None
    try:
        debrief = build_debrief(segments, roles_inferred=not roles_known)
    except Exception as exc:  # le débrief est un plus quand le rapport visuel existe ; sinon on le dit clairement
        print(f"Débrief indisponible : {exc!r}")
    if report is None and debrief is None:
        raise NoUsableContent("Ni visage exploitable ni texte suffisant : impossible de produire un rapport (enregistrement trop court ou sans parole ?).")

    transcript, convo = transcript_lines(segments), conversation_metrics(segments)
    print("[7/7] Rendu du rapport…")
    if report is not None:
        html_path = render_html(
            report, run_dir / "report.html", subject_name=subject_name, context=context, duration_min=duration / 60,
            behavior_profile=behavior_profile, transcript=transcript, convo=convo, debrief=debrief,
        )
    else:
        html_path = render_debrief_only(
            debrief, run_dir / "report.html", subject_name=subject_name, context=context, duration_min=duration / 60,
            visual_note=visual_note, transcript=transcript, convo=convo,
        )
    try:
        render_pdf(html_path)
    except Exception as exc:
        print(f"PDF indisponible : {exc!r}")
    print(f"Terminé : {html_path}")
    return html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse comportementale d'une vidéo")
    parser.add_argument("video", type=Path, help="chemin de la vidéo (doit montrer le prospect/interlocuteur externe, jamais un salarié du client)")
    parser.add_argument("--subject", default="Interlocuteur analysé", help="nom du prospect/interlocuteur pour le rapport")
    parser.add_argument("--context", default="entretien de vente B2B (interlocuteur externe)", help="contexte de l'enregistrement")
    args = parser.parse_args()
    run(args.video, args.subject, args.context)
