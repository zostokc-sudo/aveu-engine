"""Extraction audio depuis une vidéo, prête pour la transcription."""

import subprocess
from pathlib import Path

FFMPEG = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"


def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )
    return out_wav


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())
