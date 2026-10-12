"""Benchmark Phase 0 trên Kaggle: ASR → dịch → nghe mù TTS. Chạy không cần tương tác.

python -m dub.bench --video "<tên file>" --lang en
Kết quả (report.json, logs/, listen/) nằm trong OUT để tải về bằng `kaggle kernels output`.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

from dub import llm, translate as tr

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get("SDD_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("SDD_OUT", "/kaggle/working"))
EDGE_VOICES = {"edge_HoaiMy": "vi-VN-HoaiMyNeural", "edge_NamMinh": "vi-VN-NamMinhNeural"}
RESULTS = {}  # số liệu của mọi bước, ghi ra report.json


def find_video(name):
    """Kaggle gắn dataset ở /kaggle/input/<slug>/ (đường dẫn có thể đổi) nên tìm theo tên file."""
    hits = list(INPUT.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Không thấy '{name}' trong {INPUT}. Có: {sorted(p.name for p in INPUT.rglob('*.mp4'))}")
    return hits[0]


def job_dir(video):
    d = OUT / "work" / re.sub(r"[^\w-]+", "_", Path(video).stem)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def run_engine(engine, *args):
    """Chạy engine trong venv riêng; trả về (JSON dòng cuối, VRAM đỉnh MB, thời gian thực)."""
    peak, stop = [0], threading.Event()

    def poll():
        while not stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                peak[0] = max(peak[0], int(r.stdout.split()[0]))
            stop.wait(0.5)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    cmd = ["bash", str(ROOT / "engines/run.sh"), engine, *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    th.join()
    log = OUT / "logs" / f"{engine}_{time.strftime('%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\nexit {p.returncode}, {wall:.1f}s\n\n--- stdout ---\n{p.stdout}\n"
                   f"--- stderr ---\n{p.stderr}", encoding="utf-8")
    if p.returncode:
        print(p.stderr[-4000:])
        raise RuntimeError(f"Engine {engine} lỗi (mã {p.returncode}), log: {log}")
    return json.loads(p.stdout.strip().splitlines()[-1]), peak[0], round(wall, 1)


def asr(video, lang, minutes, model):
    d = job_dir(video)
    wav = d / "audio16k.wav"
    ffmpeg("-i", find_video(video), "-t", minutes * 60, "-vn", "-ac", 1, "-ar", 16000, wav)
    audio_s = duration(wav)
    print(f"ASR {audio_s / 60:.1f} phút audio bằng {model}...", flush=True)
    info, vram, wall = run_engine("asr", wav, d / "asr.json", "--lang", lang, "--model", model)
    segs = json.loads((d / "asr.json").read_text(encoding="utf-8"))["segments"]
    no_ts = sum(1 for s in segs for w in s["words"] if w["s"] is None)
    RESULTS["asr"] = {"video": video, "model": model, "audio_s": round(audio_s, 1), "wall_s": wall,
                      "rtf": round(wall / audio_s, 3), "vram_peak_mb": vram,
                      "words_without_timestamp": f"{no_ts}/{sum(len(s['words']) for s in segs)}", **info}


def translate(video, n, voice_rate=5.0):
    d = job_dir(video)
    asr_json = json.loads((d / "asr.json").read_text(encoding="utf-8"))
    segs = asr_json["segments"][:n]
    calls0, t0 = llm.calls, time.time()
    print(f"Dịch {len(segs)} câu bằng Gemini...", flush=True)
    ctx = tr.context_pass(segs, asr_json["language"])
    rows = tr.translate(segs, asr_json["language"], ctx, voice_rate=voice_rate)
    for s, r in zip(segs, rows):
        r["src"] = s["text"]
    over = [r for r in rows if tr.syllables(r["dub"]) > r["max_syll"]]
    (d / "translated.json").write_text(json.dumps({"context": ctx, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    RESULTS["translate"] = {"model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), "segments": len(rows),
                            "requests": llm.calls - calls0, "wall_s": round(time.time() - t0, 1),
                            "over_budget": f"{len(over)}/{len(rows)}"}


def _auto_ref(video, out):
    """Giọng mẫu = câu 6–12 giây đầu tiên của chính video (clone giọng người nói gốc)."""
    segs = json.loads((job_dir(video) / "asr.json").read_text(encoding="utf-8"))["segments"]
    pick = next((s for s in segs if 6 <= s["end"] - s["start"] <= 12), max(segs, key=lambda s: s["end"] - s["start"]))
    ffmpeg("-ss", pick["start"], "-to", min(pick["end"], pick["start"] + 12), "-i", find_video(video),
           "-vn", "-ac", 1, "-ar", 22050, out)
    return out


def tts(video, n):
    """n câu × 3 giọng (viXTTS + 2 edge-tts). File nghe đặt tên theo nhãn ẩn danh X/Y/Z."""
    d = job_dir(video)
    rows = json.loads((d / "translated.json").read_text(encoding="utf-8"))["rows"]
    # câu dài vừa (8–30 âm tiết): câu quá ngắn không đủ để chấm giọng
    picks = ([r for r in rows if 8 <= tr.syllables(r["dub"]) <= 30] or rows)[:n]
    raw = d / "tts"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir()

    ref = _auto_ref(video, d / "ref.wav")
    items = [{"id": r["id"], "text": r["dub"], "out": str(raw / f"viXTTS_{r['id']}.wav")} for r in picks]
    (d / "tts_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print("viXTTS...", flush=True)
    info, vram, _ = run_engine("tts", d / "tts_items.json", ref)
    stats = {"viXTTS": {"load_s": info["load_s"], "vram_peak_mb": vram,
                        "rtf": round(sum(i["synth_s"] for i in info["items"]) / sum(i["dur"] for i in info["items"]), 3)}}

    for name, voice in EDGE_VOICES.items():
        t0 = time.time()
        for r in picks:
            subprocess.run(["edge-tts", "--voice", voice, "--text", r["dub"],
                            "--write-media", str(raw / f"{name}_{r['id']}.mp3")], check=True, capture_output=True)
        stats[name] = {"wall_s": round(time.time() - t0, 1)}

    syl = sum(tr.syllables(r["dub"]) for r in picks)
    for name in stats:
        stats[name]["voice_rate"] = round(syl / sum(duration(f) for f in raw.glob(f"{name}_*")), 2)  # âm tiết/giây

    listen = OUT / "listen"
    shutil.rmtree(listen, ignore_errors=True)
    listen.mkdir()
    shutil.copy(ref, listen / "0_giong_mau_goc.wav")
    labels = dict(zip("XYZ", random.sample(list(stats), 3)))
    for label, name in labels.items():
        for k, r in enumerate(picks, 1):
            src = next(raw.glob(f"{name}_{r['id']}.*"))
            shutil.copy(src, listen / f"{label}_{k:02d}{src.suffix}")
    (listen / "cau.txt").write_text("\n".join(f"{k:02d}. {r['dub']}" for k, r in enumerate(picks, 1)), encoding="utf-8")
    RESULTS["tts"] = {"stats": stats, "blind_key": labels}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--n-seg", type=int, default=50)
    ap.add_argument("--n-tts", type=int, default=10)
    a = ap.parse_args(argv)
    steps = [("asr", lambda: asr(a.video, a.lang, a.minutes, a.model)),
             ("translate", lambda: translate(a.video, a.n_seg)),
             ("tts", lambda: tts(a.video, a.n_tts))]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:  # bước sau cần kết quả bước trước nên dừng, nhưng vẫn ghi report
            RESULTS[f"{name}_error"] = repr(e)
            traceback.print_exc()
            break
    (OUT / "report.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(RESULTS, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
