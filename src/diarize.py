"""Séparation des voix (qui parle quand) pour ne jamais confondre le commercial et le prospect.

Principe : on découpe l'audio en tours de parole (2 locuteurs), puis on identifie lequel est la
personne FILMÉE en comparant, fenêtre par fenêtre, l'activité de chaque voix au mouvement des
lèvres du visage (visual_speech_ratio). Le locuteur dont la voix colle le mieux aux lèvres est
la personne filmée. Si la corrélation est trop faible pour trancher, on ne devine pas : le
pipeline retombe sur son comportement d'avant (aucune étiquette, texte mélangé).

Moteur : sherpa-onnx (ONNX, CPU, gratuit, aucun compte). Si la bibliothèque ou les modèles
manquent, `diarize` renvoie [] et l'analyse continue sans séparation."""

import statistics
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
SEGMENTATION_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_MODEL = MODELS_DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"  # voix anglaises/françaises ; validé sur 2 voix alternées

_MIN_CORRELATION = 0.12   # en dessous, la voix ne suit pas les lèvres : on ne tranche pas
_MIN_MARGIN = 0.08        # écart minimal entre le meilleur et le second locuteur


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def diarize(wav_path: Path, num_speakers: int = 2) -> list[Turn]:
    try:
        import sherpa_onnx
    except ImportError:
        return []
    if not (SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()):
        return []
    try:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION_MODEL)),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING_MODEL)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=0.5),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        engine = sherpa_onnx.OfflineSpeakerDiarization(config)
        with wave.open(str(wav_path), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if w.getframerate() != engine.sample_rate:
                return []
        result = engine.process(audio).sort_by_start_time()
        return [Turn(float(s.start), float(s.end), int(s.speaker)) for s in result]
    except Exception as exc:  # la séparation est un plus : jamais bloquer l'audit pour elle
        print(f"Séparation des voix indisponible : {exc!r}")
        return []


def _overlap(turns: list[Turn], speaker: int, start: float, end: float) -> float:
    return sum(max(0.0, min(t.end, end) - max(t.start, start)) for t in turns if t.speaker == speaker)


def _correlation(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or statistics.pstdev(a) < 1e-9 or statistics.pstdev(b) < 1e-9:
        return 0.0
    return statistics.correlation(a, b)


def filmed_speaker(turns: list[Turn], buckets) -> int | None:
    """Locuteur dont la voix suit le mieux le mouvement des lèvres de la personne filmée."""
    usable = [b for b in buckets if b.face_coverage >= 0.3]
    speakers = sorted({t.speaker for t in turns})
    if len(speakers) < 2 or not usable:
        return None
    lips = [b.visual_speech_ratio for b in usable]
    scores = {
        s: _correlation([_overlap(turns, s, b.start, b.end) / max(b.end - b.start, 1e-6) for b in usable], lips)
        for s in speakers
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_c), (_, second_c) = ranked[0], ranked[1]
    if best_c < _MIN_CORRELATION or best_c - second_c < _MIN_MARGIN:
        return None
    return best


def label_segments(segments, turns: list[Turn], filmed: int | None) -> bool:
    """Étiquette chaque segment 'filmed' (la personne à l'écran) ou 'other' (son interlocuteur).
    Renvoie True si l'étiquetage a eu lieu."""
    if filmed is None or not turns:
        return False
    for seg in segments:
        per_speaker: dict[int, float] = {}
        for t in turns:
            ov = max(0.0, min(t.end, seg.end) - max(t.start, seg.start))
            if ov > 0:
                per_speaker[t.speaker] = per_speaker.get(t.speaker, 0.0) + ov
        if not per_speaker:
            seg.speaker = None
            continue
        seg.speaker = "filmed" if max(per_speaker, key=per_speaker.get) == filmed else "other"
    return True
