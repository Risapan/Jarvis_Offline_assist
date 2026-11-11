#!/usr/bin/env bash
# Download the Kokoro TTS model files into ./models
set -euo pipefail
cd "$(dirname "$0")/../models"
BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -f "$f" ] && echo "✓ $f" || { echo "↓ $f"; curl -L --fail -o "$f" "$BASE/$f"; }
done
