"""Extraction de signaux comportementaux (visage + posture) frame par frame.

Repose sur MediaPipe Face Landmarker (52 blendshapes ARKit/FACS) et Pose
Landmarker. Les signaux sont agrégés par fenêtre temporelle (`bucket_seconds`)
pour être croisés ensuite avec la transcription.

Principe de précision central : un signal facial brut (ex. tension de
mâchoire) est ambigu en soi — parler active naturellement la mâchoire, sans
lien avec une émotion. On garde donc, en plus des moyennes, le PIC et la
VARIABILITÉ de chaque signal par fenêtre (un pic bref compte plus qu'une
moyenne lissée), et plusieurs blendshapes complémentaires pour désambiguïser
(ex. cheek_raise + smile = sourire authentique de Duchenne ; browDown seul
peut être de la concentration, browDown + eyeSquint + lipPress = tension).
La désambiguïsation "parle/ne parle pas" et le calibrage par rapport à la
ligne de base personnelle se font en aval, dans report_generator.merge_timeline
— cette machine n'a que les données visuelles, pas la transcription.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
SAMPLE_FPS = 5.0


@dataclass
class FrameSignals:
    t: float
    smile: float = 0.0
    cheek_raise: float = 0.0  # marqueur du sourire authentique (Duchenne) combiné au smile
    mouth_frown: float = 0.0  # déplaisir/moue, distinct de la tension mâchoire
    brow_tension: float = 0.0  # sourcils abaissés — concentration OU tension, ambigu seul
    inner_brow_raise: float = 0.0  # sourcils intérieurs relevés — inquiétude/tristesse
    jaw_tension: float = 0.0
    jaw_open: float = 0.0  # ouverture de la bouche — sert à détecter l'activité labiale (parle/ne parle pas)
    eye_wide: float = 0.0  # surprise/alerte
    blink: bool = False
    gaze_offset: float = 0.0  # 0 = regarde la caméra, 1 = regard très détourné
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    hand_to_face: bool = False
    torso_lean: float = 0.0  # écart à la verticale, degrés
    wrist_velocity: float = 0.0  # déplacement des mains / seconde (fidgeting)
    face_detected: bool = False
    pose_detected: bool = False
    multi_face: bool = False  # >1 visage détecté dans le cadre à cet instant


@dataclass
class BucketSignals:
    start: float
    end: float
    smile_avg: float = 0.0
    smile_peak: float = 0.0
    cheek_raise_avg: float = 0.0
    mouth_frown_avg: float = 0.0
    brow_tension_avg: float = 0.0
    brow_tension_peak: float = 0.0
    brow_tension_std: float = 0.0
    inner_brow_raise_avg: float = 0.0
    jaw_tension_avg: float = 0.0
    jaw_tension_peak: float = 0.0
    eye_wide_avg: float = 0.0
    blink_rate_per_min: float = 0.0
    gaze_aversion_avg: float = 0.0
    gaze_aversion_std: float = 0.0
    head_movement: float = 0.0  # variance yaw+pitch = agitation de la tête
    hand_to_face_ratio: float = 0.0
    torso_lean_avg: float = 0.0
    fidget_score: float = 0.0  # vitesse moyenne des mains (agitation)
    fidget_peak: float = 0.0
    face_coverage: float = 0.0  # % de frames avec visage détecté
    multi_face_ratio: float = 0.0  # % de frames avec >1 visage — risque de mélanger 2 personnes
    visual_speech_ratio: float = 0.0  # fraction du temps où les lèvres bougent (parle probablement, indépendant de l'audio)


def _blendshape_map(result) -> dict[str, float]:
    if not result.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def _analyze_face(landmarker: FaceLandmarker, frame_rgb, t_ms: int) -> tuple[float, ...]:
    from mediapipe import Image, ImageFormat

    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, t_ms)

    if not result.face_landmarks:
        # smile, cheek_raise, mouth_frown, brow, inner_brow, jaw, jaw_open, eye_wide, blink, gaze, yaw, pitch, face_ok, multi_face
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, False, False

    multi_face = len(result.face_landmarks) > 1
    bs = _blendshape_map(result)
    smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2
    cheek_raise = (bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)) / 2
    mouth_frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2
    brow_tension = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2
    inner_brow_raise = bs.get("browInnerUp", 0.0)
    jaw_tension = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0) + bs.get("jawForward", 0.0)) / 3
    jaw_open = bs.get("jawOpen", 0.0)
    eye_wide = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2
    blink = ((bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2) > 0.5
    gaze = (
        bs.get("eyeLookOutLeft", 0.0) + bs.get("eyeLookOutRight", 0.0)
        + bs.get("eyeLookInLeft", 0.0) + bs.get("eyeLookInRight", 0.0)
    ) / 4

    yaw, pitch = 0.0, 0.0
    if result.facial_transformation_matrixes:
        m = result.facial_transformation_matrixes[0]
        yaw = math.degrees(math.atan2(-m[2][0], math.sqrt(m[2][1] ** 2 + m[2][2] ** 2)))
        pitch = math.degrees(math.atan2(m[2][1], m[2][2]))

    return smile, cheek_raise, mouth_frown, brow_tension, inner_brow_raise, jaw_tension, jaw_open, eye_wide, blink, gaze, yaw, pitch, True, multi_face


def _analyze_pose(landmarker: PoseLandmarker, frame_rgb, t_ms: int) -> tuple[float, bool, tuple[float, float] | None]:
    from mediapipe import Image, ImageFormat

    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, t_ms)

    if not result.pose_landmarks:
        return 0.0, False, None

    lm = result.pose_landmarks[0]
    # indices MediaPipe Pose : 11/12 épaules, 23/24 hanches, 15/16 poignets, 0 nez
    l_sh, r_sh = lm[11], lm[12]
    l_hip, r_hip = lm[23], lm[24]
    shoulder_mid = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
    hip_mid = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    torso_lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    nose = lm[0]
    l_wrist, r_wrist = lm[15], lm[16]
    hand_near_face = any(
        math.hypot(w.x - nose.x, w.y - nose.y) < 0.15 for w in (l_wrist, r_wrist)
    )

    wrist_pos = ((l_wrist.x + r_wrist.x) / 2, (l_wrist.y + r_wrist.y) / 2)
    return torso_lean, hand_near_face, wrist_pos


def analyze_video(video_path: Path) -> list[FrameSignals]:
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELS_DIR / "face_landmarker.task"),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=2,  # detecte jusqu'a 2 visages pour repérer les cadrages a deux personnes (voir multi_face)
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODELS_DIR / "pose_landmarker.task"),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, round(native_fps / SAMPLE_FPS))

    frames: list[FrameSignals] = []
    prev_wrist: tuple[float, float] | None = None
    prev_t = 0.0

    with FaceLandmarker.create_from_options(face_options) as face_lm, \
         PoseLandmarker.create_from_options(pose_options) as pose_lm:

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue

            t = idx / native_fps
            t_ms = int(t * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            (smile, cheek_raise, mouth_frown, brow, inner_brow, jaw, jaw_open, eye_wide,
             blink, gaze, yaw, pitch, face_ok, multi_face) = _analyze_face(face_lm, frame_rgb, t_ms)
            torso_lean, hand_to_face, wrist_pos = _analyze_pose(pose_lm, frame_rgb, t_ms)

            wrist_velocity = 0.0
            if wrist_pos is not None and prev_wrist is not None and t > prev_t:
                dist = math.hypot(wrist_pos[0] - prev_wrist[0], wrist_pos[1] - prev_wrist[1])
                wrist_velocity = dist / (t - prev_t)

            fs = FrameSignals(
                t=t, smile=smile, cheek_raise=cheek_raise, mouth_frown=mouth_frown,
                brow_tension=brow, inner_brow_raise=inner_brow, jaw_tension=jaw, jaw_open=jaw_open,
                eye_wide=eye_wide, blink=blink, gaze_offset=gaze, head_yaw=yaw, head_pitch=pitch,
                hand_to_face=hand_to_face, torso_lean=torso_lean,
                wrist_velocity=wrist_velocity,
                face_detected=face_ok, pose_detected=wrist_pos is not None, multi_face=multi_face,
            )
            frames.append(fs)

            if wrist_pos is not None:
                prev_wrist = wrist_pos
                prev_t = t
            idx += 1

    cap.release()
    return frames


def bucketize(frames: list[FrameSignals], bucket_seconds: float = 5.0) -> list[BucketSignals]:
    if not frames:
        return []

    duration = frames[-1].t
    n_buckets = int(duration // bucket_seconds) + 1
    buckets: list[BucketSignals] = []

    for i in range(n_buckets):
        b_start, b_end = i * bucket_seconds, (i + 1) * bucket_seconds
        chunk = [f for f in frames if b_start <= f.t < b_end]
        if not chunk:
            continue

        face_frames = [f for f in chunk if f.face_detected]
        pose_frames = [f for f in chunk if f.pose_detected]
        n = len(chunk)
        blink_count = sum(1 for f in chunk if f.blink)
        duration_min = bucket_seconds / 60.0

        yaws = [f.head_yaw for f in face_frames]
        pitches = [f.head_pitch for f in face_frames]
        head_movement = (max(yaws) - min(yaws) if yaws else 0.0) + (max(pitches) - min(pitches) if pitches else 0.0)

        smiles = [f.smile for f in face_frames]
        brows = [f.brow_tension for f in face_frames]
        jaws = [f.jaw_tension for f in face_frames]
        gazes = [f.gaze_offset for f in face_frames]
        wrist_vels = [f.wrist_velocity for f in pose_frames]
        visual_speech_ratio = _visual_speech_ratio(face_frames)

        buckets.append(BucketSignals(
            start=b_start,
            end=b_end,
            smile_avg=_avg(smiles),
            smile_peak=_peak(smiles),
            cheek_raise_avg=_avg([f.cheek_raise for f in face_frames]),
            mouth_frown_avg=_avg([f.mouth_frown for f in face_frames]),
            brow_tension_avg=_avg(brows),
            brow_tension_peak=_peak(brows),
            brow_tension_std=_std(brows),
            inner_brow_raise_avg=_avg([f.inner_brow_raise for f in face_frames]),
            jaw_tension_avg=_avg(jaws),
            jaw_tension_peak=_peak(jaws),
            eye_wide_avg=_avg([f.eye_wide for f in face_frames]),
            blink_rate_per_min=blink_count / duration_min if duration_min else 0.0,
            gaze_aversion_avg=_avg(gazes),
            gaze_aversion_std=_std(gazes),
            head_movement=head_movement,
            hand_to_face_ratio=sum(1 for f in chunk if f.hand_to_face) / n,
            torso_lean_avg=_avg([f.torso_lean for f in chunk if f.pose_detected]),
            fidget_score=_avg(wrist_vels),
            fidget_peak=_peak(wrist_vels),
            face_coverage=len(face_frames) / n,
            multi_face_ratio=sum(1 for f in chunk if f.multi_face) / n,
            visual_speech_ratio=visual_speech_ratio,
        ))

    return buckets


# Si plus de 10% des fenêtres montrent 2 visages ou plus, il y a un vrai
# risque d'avoir mélangé les signaux de deux personnes différentes — ou pire,
# d'avoir capté le salarié en plus du prospect, ce que notre conformité
# AI Act interdit. On préfère prévenir plutôt que produire un rapport ambigu.
_MULTI_FACE_WARNING_THRESHOLD = 0.10


def check_multi_face_warning(buckets: list[BucketSignals]) -> str | None:
    """Retourne un message d'avertissement si la vidéo semble montrer plusieurs
    personnes simultanément une part significative du temps — condition pour
    laquelle l'analyse par fenêtre risque de mélanger deux visages différents,
    et où la conformité (jamais analyser le salarié) ne peut plus être garantie."""
    usable = [b for b in buckets if b.face_coverage >= 0.3]
    if not usable:
        return None
    ratio = _avg([b.multi_face_ratio for b in usable])
    if ratio > _MULTI_FACE_WARNING_THRESHOLD:
        return (
            f"{ratio*100:.0f}% des fenêtres analysées montrent au moins deux "
            "visages en même temps. L'analyse ne garantit pas de suivre la "
            "même personne d'une fenêtre à l'autre dans ce cas — recadrez la "
            "vidéo pour ne montrer QUE l'interlocuteur externe avant de "
            "relancer, sans quoi le rapport pourrait mélanger deux personnes "
            "ou, pire, capter votre salarié."
        )
    return None


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def buckets_from_dicts(dicts: list[dict]) -> list[BucketSignals]:
    """Reconstruit des BucketSignals depuis le JSON produit par vision_worker.py."""
    return [BucketSignals(**d) for d in dicts]


def _peak(values: list[float]) -> float:
    return max(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _avg(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# Delta minimal de jawOpen entre deux frames consécutives pour compter comme
# "activité labiale" (probablement en train de parler). Calibré sur un
# enregistrement réel (delta moyen observé ~0.018 pendant la parole, une
# première valeur de 0.05 s'est révélée bien trop stricte et sous-détectait
# massivement) — à affiner encore avec plus de données réelles.
_LIP_ACTIVITY_THRESHOLD = 0.015


def _visual_speech_ratio(face_frames: list[FrameSignals]) -> float:
    """Fraction du temps où les lèvres de la personne FILMÉE bougent —
    indépendant de l'audio. Sert à vérifier, en croisant avec speech_ratio
    (audio), si c'est vraiment elle qui parle : si l'audio capte une voix
    mais que ses lèvres ne bougent pas, la voix entendue est probablement
    celle de l'autre partie de l'appel (transcription non diarisée, voir
    transcribe.py)."""
    if len(face_frames) < 2:
        return 0.0
    deltas = [
        abs(b.jaw_open - a.jaw_open)
        for a, b in zip(face_frames, face_frames[1:])
    ]
    active = sum(1 for d in deltas if d > _LIP_ACTIVITY_THRESHOLD)
    return active / len(deltas)
