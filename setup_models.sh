#!/bin/bash
set -e
mkdir -p models
curl -sL -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -sL -o models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
echo "Modèles téléchargés dans models/"

# Séparation des voix (sherpa-onnx) : segmentation pyannote + empreinte de voix, tous deux gratuits et sans compte
if [ ! -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  curl -sL -o models/seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf models/seg.tar.bz2 -C models && rm -f models/seg.tar.bz2
fi
if [ ! -f models/wespeaker_en_voxceleb_resnet34_LM.onnx ]; then
  curl -sL -o models/wespeaker_en_voxceleb_resnet34_LM.onnx "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
fi
echo "Modèles de séparation des voix prêts."
