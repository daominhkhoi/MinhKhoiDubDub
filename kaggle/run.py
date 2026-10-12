"""Kernel Kaggle: clone repo, cài engine, chạy benchmark Phase 0.

Đẩy lên và chạy từ laptop:  kaggle kernels push -p kaggle
Lấy kết quả:                kaggle kernels output khiominhk19fptuhcm/superdubdub-bench -p <thư mục>
"""
import os
import subprocess
import sys

BRANCH = "feat/phase0-notebook"
VIDEO = "SQL Server 2022 - Set Up A -Dummy- Database.mp4"
REPO = "/tmp/repo"  # ngoài /kaggle/working để code không lẫn vào output

subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH,
                "https://github.com/daominhkhoi/MinhKhoiDubDub.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True)
subprocess.run(["bash", f"{REPO}/engines/setup.sh", "asr", "tts"], check=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
except Exception as e:  # chưa gắn secret: ASR vẫn chạy, bước dịch sẽ báo lỗi trong report
    print("Chưa đọc được secret GEMINI_API_KEY:", repr(e))

sys.path.insert(0, REPO)
from dub import bench  # noqa: E402

bench.main(["--video", VIDEO, "--lang", "en", "--minutes", "5"])
