"""Transcription avec timestamps mot-à-mot via mlx-whisper (local, Apple Silicon).

LIMITE CONNUE : pas de diarisation (distinction des locuteurs). Si la piste
audio capte les deux côtés de l'appel (cas fréquent des enregistrements Zoom/
Teams non isolés par participant), le texte transcrit peut mélanger les mots
du commercial et de l'interlocuteur externe pendant une même fenêtre. Voir la
mise en garde correspondante dans report_generator.SYSTEM_PROMPT — c'est géré
en aval par la prudence du prompt, pas ici. Une vraie diarisation (ex.
pyannote.audio) réglerait ça proprement mais n'est pas encore implémentée."""

from dataclasses import dataclass
from pathlib import Path

try:
    import mlx_whisper  # Apple Silicon (ton Mac)
except ImportError:
    mlx_whisper = None  # serveur Linux : on bascule sur faster-whisper (CPU)


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


def _transcribe_cpu(wav_path: Path) -> list[Segment]:
    from faster_whisper import WhisperModel

    import wave

    import numpy as np

    # extract_audio produit déjà un WAV 16 kHz mono 16 bits : on le charge directement
    # (évite la dépendance à PyAV, dont la version varie selon les installations).
    with wave.open(str(wav_path), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    model = WhisperModel("small", device="cpu", compute_type="int8")
    raw, _ = model.transcribe(audio, word_timestamps=True)
    return [
        Segment(
            text=seg.text.strip(), start=seg.start, end=seg.end,
            words=[Word(text=w.word.strip(), start=w.start, end=w.end) for w in (seg.words or [])],
        )
        for seg in raw
    ]


def transcribe(wav_path: Path, model: str = "mlx-community/whisper-large-v3-turbo") -> list[Segment]:
    if mlx_whisper is None:
        return _transcribe_cpu(wav_path)
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model,
        word_timestamps=True,
    )

    segments: list[Segment] = []
    for seg in result["segments"]:
        words = [
            Word(text=w["word"].strip(), start=w["start"], end=w["end"])
            for w in seg.get("words", [])
        ]
        segments.append(
            Segment(text=seg["text"].strip(), start=seg["start"], end=seg["end"], words=words)
        )
    return segments
