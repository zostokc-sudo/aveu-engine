"""Extraction de signaux prosodiques (hauteur de voix, énergie, régularité)
directement depuis l'audio — complète l'analyse visuelle et la transcription
texte, qui ne capturent ni l'une ni l'autre COMMENT une phrase est dite.

La voix porte des signaux indépendants du contenu verbal : une hausse de la
hauteur (pitch) associée à une baisse de la variabilité peut marquer une
tension dans la voix ; une énergie qui chute brutalement en milieu de phrase
peut marquer une hésitation ; un débit qui devient monocorde après un pic de
variation peut marquer un repli. Comme pour le visage, aucun de ces signaux
n'est probant seul — ils se combinent et se calibrent sur la ligne de base
vocale de la personne, exactement comme les signaux visuels.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# hop_length généreux : on agrège sur des fenêtres de 5s, pas besoin d'une
# résolution fine. pyin (probabiliste, précis) est ~15x trop lent pour un
# appel d'1h — yin (déterministe) est largement suffisant une fois moyenné.
HOP_LENGTH = 2048


@dataclass
class AudioBucketSignals:
    start: float
    end: float
    pitch_avg_hz: float = 0.0
    pitch_std_hz: float = 0.0
    energy_avg: float = 0.0
    energy_peak: float = 0.0
    energy_std: float = 0.0
    voiced_ratio: float = 0.0  # fraction du temps où une voix est détectée (parle vs silence/bruit)


def analyze_audio(wav_path: Path, bucket_seconds: float = 5.0) -> list[AudioBucketSignals]:
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(y) == 0:
        return []

    f0 = librosa.yin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=HOP_LENGTH,
    )
    f0_times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    # yin ne renvoie pas de flag "voisé" — une énergie quasi nulle sur la
    # fenêtre correspondante indique un silence plutôt qu'une vraie hauteur.
    frame_rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=2048)[0]
    n = min(len(f0), len(frame_rms))
    voiced_flag = frame_rms[:n] > (0.15 * np.max(frame_rms)) if len(frame_rms) else np.array([])
    f0, f0_times = f0[:n], f0_times[:n]

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH)

    duration = len(y) / sr
    n_buckets = int(duration // bucket_seconds) + 1
    buckets = []

    for i in range(n_buckets):
        b_start, b_end = i * bucket_seconds, (i + 1) * bucket_seconds
        f0_mask = (f0_times >= b_start) & (f0_times < b_end)
        rms_mask = (rms_times >= b_start) & (rms_times < b_end)

        f0_chunk = f0[f0_mask]
        voiced_chunk = voiced_flag[f0_mask] if len(voiced_flag) else np.zeros_like(f0_chunk, dtype=bool)
        # yin renvoie toujours une estimation, même sur du silence/bruit —
        # on ne garde que les frames marquées voisées (énergie suffisante).
        f0_voiced = f0_chunk[voiced_chunk] if len(voiced_chunk) else np.array([])
        rms_chunk = rms[rms_mask]

        buckets.append(AudioBucketSignals(
            start=b_start, end=b_end,
            pitch_avg_hz=float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0,
            pitch_std_hz=float(np.std(f0_voiced)) if len(f0_voiced) > 1 else 0.0,
            energy_avg=float(np.mean(rms_chunk)) if len(rms_chunk) else 0.0,
            energy_peak=float(np.max(rms_chunk)) if len(rms_chunk) else 0.0,
            energy_std=float(np.std(rms_chunk)) if len(rms_chunk) > 1 else 0.0,
            voiced_ratio=float(np.mean(voiced_chunk)) if len(voiced_chunk) else 0.0,
        ))

    return buckets


def audio_buckets_by_start(buckets: list[AudioBucketSignals]) -> dict[float, AudioBucketSignals]:
    return {b.start: b for b in buckets}
