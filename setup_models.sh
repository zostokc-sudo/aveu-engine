#!/bin/bash
set -e
mkdir -p models
curl -sL -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -sL -o models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
echo "Modèles téléchargés dans models/"
