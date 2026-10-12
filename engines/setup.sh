#!/usr/bin/env bash
# Cài môi trường trên Kaggle. Mỗi engine một venv Python 3.11 để không đụng torch có sẵn của image.
# Cách dùng: engines/setup.sh asr tts
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { apt-get -qq update && apt-get -qq install -y ffmpeg; }
pip install -q uv edge-tts requests
for e in "$@"; do
  # chỉ bỏ qua khi lần cài trước chạy hết; cài dở (bị ngắt) thì làm lại từ đầu
  if [ -f "engines/$e/.venv/.installed" ]; then
    echo "[$e] đã cài trong phiên này, bỏ qua"
    continue
  fi
  start=$SECONDS
  uv venv -q --clear --python 3.11 "engines/$e/.venv"
  uv pip install -q --python "engines/$e/.venv/bin/python" -r "engines/$e/requirements.txt"
  touch "engines/$e/.venv/.installed"
  echo "[$e] cài xong sau $((SECONDS - start)) giây"
done
