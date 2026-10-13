"""Kernel Kaggle: clone repo, cài engine, chạy benchmark Phase 0.

Đẩy lên và chạy từ laptop:  kaggle kernels push -p kaggle
Lấy kết quả:                kaggle kernels output khiominhk19fptuhcm/superdubdub-bench -p <thư mục>
"""
import os
import subprocess
import sys
import time

BRANCH = "feat/phase0-notebook"
VIDEO = "SQL Server 2022 - Set Up A -Dummy- Database.mp4"
REPO = "/tmp/repo"  # ngoài /kaggle/working để code không lẫn vào output

# đọc secret trước khi cài đặt: lỗi thì biết ngay, không đợi 5 phút
print("Có token dịch vụ secrets:", bool(os.environ.get("KAGGLE_USER_SECRETS_TOKEN")))
for attempt in range(3):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
        print("Đã đọc secret GEMINI_API_KEY")
        break
    except Exception as e:  # chưa đọc được: ASR vẫn chạy, bước dịch sẽ báo lỗi trong report
        print(f"Lần {attempt + 1}: chưa đọc được secret GEMINI_API_KEY:", repr(e))
        time.sleep(10)

subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH,
                "https://github.com/daominhkhoi/MinhKhoiDubDub.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])  # CPU thì báo lỗi, bỏ qua
subprocess.run(["bash", f"{REPO}/engines/setup.sh", "asr", "tts"], check=True)

sys.path.insert(0, REPO)
from dub import bench  # noqa: E402

# shortcut: đang chạy CPU vì hết hạn mức GPU tuần này (làm mới 17/10); có GPU lại thì về large-v3, 5 phút
bench.main(["--video", VIDEO, "--lang", "en", "--minutes", "3", "--model", "large-v3-turbo"])
