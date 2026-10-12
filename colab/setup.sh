#!/usr/bin/env bash
# Cài môi trường trên Colab. Mỗi engine một venv Python 3.11 để không đụng torch có sẵn của Colab.
# Cách dùng: colab/setup.sh asr tts
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q uv edge-tts requests
for e in "$@"; do
  if [ -x "engines/$e/.venv/bin/python" ]; then
    echo "[$e] đã cài trong phiên này, bỏ qua"
    continue
  fi
  start=$SECONDS
  uv venv -q --python 3.11 "engines/$e/.venv"
  uv pip install -q --python "engines/$e/.venv/bin/python" -r "engines/$e/requirements.txt"
  echo "[$e] cài xong sau $((SECONDS - start)) giây"
done
