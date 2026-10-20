"""Orchestrateur : vidéo -> transcription + signaux comportementaux -> rapport HTML."""

import argparse
import json
import time
from pathlib import Path

from audio_analysis import analyze_audio
from debrief import NotASalesCall, build_debrief
from conversation import conversation_metrics, transcript_lines
from delivery import delivery_metrics
from image_env import analyze_image
from style import build_style, style_text_available
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


def run(video_path: Path, subject_name: str, context: str, source_url: str | None = None,
        filmed_role: str = "prospect", focus: tuple | list = (), analyze_my_image: bool = False,
        benchmark: dict | None = None) -> Path:
    """filmed_role : 'prospect' (cas normal : jauges de réactions sur le prospect) ou 'closer' (auto-évaluation : pas
    d'analyse de réactions du visage, seulement le texte, la voix, le vocabulaire et, si demandé, l'image/le décor)."""
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
        warning = check_multi_face_warning(buckets) if filmed_role == "prospect" else None
        if warning:
            visual_ok, visual_note = False, _no_face_note("Plusieurs visages apparaissent à l'écran en même temps : impossible de suivre la même personne sans risque de confusion.")

    turns = diarize(wav_path)
    roles_known = label_segments(segments, turns, filmed_speaker(turns, buckets) if visual_ok else None, filmed_role)
    print("Voix séparées : la personne filmée est identifiée." if roles_known else "Voix non étiquetées : les rôles seront déduits du texte.")

    if visual_ok and filmed_role == "closer":
        visual_note = ("Vous êtes à l'écran : par principe, Aveu n'analyse pas vos expressions du visage. "
                       "Ce rapport porte sur vos mots, votre voix et, si vous l'avez demandé, votre image.")
    if visual_ok and filmed_role == "prospect":
        print("[5/7] Rapport sur les réactions du prospect…")
        report = generate_report(segments, buckets, context=context, audio_buckets=audio_buckets)
        save_report_json(report, run_dir / "report.json")
        behavior_profile = compute_behavior_profile(buckets)

    print("[6/7] Débrief du closer (texte)…")
    debrief, not_sales = None, None
    try:
        debrief = build_debrief(segments, roles_inferred=not roles_known, focus=focus)
    except NotASalesCall as exc:
        not_sales = str(exc)
        print(f"Débrief non applicable : {exc}")
    except Exception as exc:  # le débrief est un plus quand le rapport visuel existe ; sinon on le dit clairement
        print(f"Débrief indisponible : {exc!r}")
    if report is None and debrief is None:
        if not_sales:
            raise NoUsableContent(not_sales + " Envoyez l'enregistrement d'un appel entre un vendeur et un prospect.")
        raise NoUsableContent("Ni visage exploitable ni texte suffisant : impossible de produire un rapport (enregistrement trop court ou sans parole ?).")

    extras = {"focus": list(focus), "benchmark": benchmark if benchmark and benchmark.get("opening") is not None else None}
    extras["delivery"] = delivery_metrics(segments) if roles_known else None
    extras["style"] = None
    try:
        if debrief is not None and style_text_available(segments, roles_known):
            time.sleep(40)  # plafond gratuit Groq par minute : on laisse la fenêtre se vider après le débrief
            extras["style"] = build_style(segments, roles_known)
    except Exception as exc:  # le style du prospect est un plus : jamais bloquant
        print(f"Style du prospect indisponible : {exc!r}")
    if analyze_my_image and filmed_role == "closer" and visual_ok:
        try:
            extras["image_env"] = analyze_image(norm_path, duration)
        except Exception as exc:
            print(f"Image indisponible : {exc!r}")

    transcript, convo = transcript_lines(segments), conversation_metrics(segments)
    if debrief is not None:  # statistiques anonymes (aucun texte) : elles servent de point de comparaison aux prochains débriefs
        d = extras.get("delivery") or {}
        (run_dir / "stats.json").write_text(json.dumps({
            "scores": {c["key"]: c["score"] for c in debrief["scorecard"]},
            "wpm": d.get("wpm"), "soft_per_10min": d.get("soft_per_10min"), "challenge_pct": d.get("challenge_pct"),
            "closer_share": (convo or {}).get("other_share"), "next_step_secured": debrief["next_step_secured"],
        }), encoding="utf-8")
    print("[7/7] Rendu du rapport…")
    if report is not None:
        html_path = render_html(
            report, run_dir / "report.html", subject_name=subject_name, context=context, duration_min=duration / 60,
            behavior_profile=behavior_profile, transcript=transcript, convo=convo, debrief=debrief,
            source_url=source_url, source_title=subject_name, extras=extras,
        )
    else:
        html_path = render_debrief_only(
            debrief, run_dir / "report.html", subject_name=subject_name, context=context, duration_min=duration / 60,
            visual_note=visual_note, transcript=transcript, convo=convo,
            source_url=source_url, source_title=subject_name, extras=extras,
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
