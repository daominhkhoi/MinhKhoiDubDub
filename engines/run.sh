#!/usr/bin/env bash
# Chạy engines/<engine>/run.py bằng venv riêng của engine đó.
# Cách dùng: engines/run.sh <engine> [tham số...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
engine="$1"
shift
py="$here/$engine/.venv/bin/python"
# ctranslate2/torch tìm cuDNN, cuBLAS trong các gói pip nvidia-* của venv, không phải trong hệ thống Colab
libs="$("$py" -c 'import glob, os, site; print(":".join(p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))')"
export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$py" "$here/$engine/run.py" "$@"
