"""Benchmark Phase 0 trên Colab: ASR → dịch → nghe mù TTS. Kết quả lưu lên Drive."""
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from dub import llm, translate as tr

ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path("/content/drive/MyDrive/SuperDubDub")
WORK = Path("/content/work")
EDGE_VOICES = {"edge_HoaiMy": "vi-VN-HoaiMyNeural", "edge_NamMinh": "vi-VN-NamMinhNeural"}
RESULTS = {}  # gom số liệu của mọi bước, lưu bằng save_report()


def job_dir(video):
    d = WORK / re.sub(r"[^\w-]+", "_", Path(video).stem)
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
    p = subprocess.run(["bash", str(ROOT / "engines/run.sh"), engine, *map(str, args)], capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    th.join()
    if p.returncode:
        print(p.stdout[-3000:], "\n--- stderr ---\n", p.stderr[-6000:])
        raise RuntimeError(f"Engine {engine} lỗi (mã {p.returncode}). Gửi đoạn log phía trên cho Claude.")
    return json.loads(p.stdout.strip().splitlines()[-1]), peak[0], round(wall, 1)


def asr(video, lang, minutes, model="large-v3"):
    src = DRIVE / "input" / video
    if not src.exists():
        raise FileNotFoundError(f"Không thấy '{video}' trong Drive/SuperDubDub/input. "
                                f"Các file có: {sorted(p.name for p in src.parent.iterdir())}")
    d = job_dir(video)
    wav = d / "audio16k.wav"
    ffmpeg("-i", src, "-t", minutes * 60, "-vn", "-ac", 1, "-ar", 16000, wav)
    audio_s = duration(wav)
    print(f"Đang nhận dạng {audio_s / 60:.1f} phút audio bằng {model} (lần đầu phải tải model)...")
    info, vram, wall = run_engine("asr", wav, d / "asr.json", "--lang", lang, "--model", model)
    segs = json.loads((d / "asr.json").read_text(encoding="utf-8"))["segments"]
    no_ts = sum(1 for s in segs for w in s["words"] if w["s"] is None)
    total_w = sum(len(s["words"]) for s in segs)
    RESULTS["asr"] = {"video": video, "model": model, "lang": info["language"], "audio_s": round(audio_s, 1),
                      "wall_s": wall, "rtf": round(wall / audio_s, 3), "vram_peak_mb": vram,
                      "segments": len(segs), "words_without_timestamp": f"{no_ts}/{total_w}", **info}
    print(json.dumps(RESULTS["asr"], ensure_ascii=False, indent=1))
    for s in segs[:8]:
        print(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['text']}")


def translate(video, n=50, voice_rate=5.0):
    d = job_dir(video)
    asr_json = json.loads((d / "asr.json").read_text(encoding="utf-8"))
    segs = asr_json["segments"][:n]
    calls0, t0 = llm.calls, time.time()
    ctx = tr.context_pass(segs, asr_json["language"])
    rows = tr.translate(segs, asr_json["language"], ctx, voice_rate=voice_rate)
    over = [r for r in rows if tr.syllables(r["dub"]) > r["max_syll"]]
    (d / "translated.json").write_text(json.dumps({"context": ctx, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    RESULTS["translate"] = {"model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), "segments": len(rows),
                            "requests": llm.calls - calls0, "wall_s": round(time.time() - t0, 1),
                            "over_budget": f"{len(over)}/{len(rows)}"}
    print(json.dumps(RESULTS["translate"], ensure_ascii=False, indent=1))
    print("\nNgữ cảnh:", ctx["summary"], "|", ctx["genre"], "|", ctx["register"])
    for s, r in zip(segs, rows):
        print(f"\n#{r['id']}  {s['text']}\n  SUB: {r['sub']}\n  DUB: {r['dub']}  ({tr.syllables(r['dub'])}/{r['max_syll']} âm tiết)")


def _auto_ref(video, out):
    """Giọng mẫu = câu 6–12 giây đầu tiên của chính video (clone giọng người nói gốc)."""
    segs = json.loads((job_dir(video) / "asr.json").read_text(encoding="utf-8"))["segments"]
    pick = next((s for s in segs if 6 <= s["end"] - s["start"] <= 12), max(segs, key=lambda s: s["end"] - s["start"]))
    ffmpeg("-ss", pick["start"], "-to", min(pick["end"], pick["start"] + 12), "-i", DRIVE / "input" / video,
           "-vn", "-ac", 1, "-ar", 22050, out)
    return out


def tts(video, n=10, voice_ref=""):
    """Tổng hợp n câu bằng viXTTS và 2 giọng edge-tts, gán nhãn ẩn danh để nghe mù."""
    from IPython.display import Audio, display

    d = job_dir(video)
    rows = json.loads((d / "translated.json").read_text(encoding="utf-8"))["rows"]
    # ưu tiên câu dài vừa (8–30 âm tiết): câu quá ngắn không đủ để chấm giọng
    pool = [r for r in rows if 8 <= tr.syllables(r["dub"]) <= 30] or rows
    picks = pool[:n]
    out = d / "tts"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()

    ref = Path(voice_ref) if voice_ref else _auto_ref(video, d / "ref.wav")
    print("Giọng mẫu cho viXTTS:", ref)
    display(Audio(str(ref)))
    items = [{"id": r["id"], "text": r["dub"], "out": str(out / f"viXTTS_{r['id']}.wav")} for r in picks]
    (d / "tts_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print("Đang chạy viXTTS (lần đầu phải tải model ~2 GB)...")
    info, vram, wall = run_engine("tts", d / "tts_items.json", ref)
    stats = {"viXTTS": {"load_s": info["load_s"], "vram_peak_mb": vram,
                        "rtf": round(sum(i["synth_s"] for i in info["items"]) / sum(i["dur"] for i in info["items"]), 3)}}

    for name, voice in EDGE_VOICES.items():
        t0 = time.time()
        for r in picks:
            subprocess.run(["edge-tts", "--voice", voice, "--text", r["dub"],
                            "--write-media", str(out / f"{name}_{r['id']}.mp3")], check=True, capture_output=True)
        stats[name] = {"wall_s": round(time.time() - t0, 1)}

    syl = sum(tr.syllables(r["dub"]) for r in picks)
    for name in stats:
        files = sorted(out.glob(f"{name}_*"))
        stats[name]["voice_rate"] = round(syl / sum(duration(f) for f in files), 2)  # âm tiết/giây, cho budgets()

    labels = dict(zip(["Giọng X", "Giọng Y", "Giọng Z"], random.sample(list(stats), 3)))
    RESULTS["tts"] = {"stats": stats, "blind_key": labels}
    (d / "blind_key.json").write_text(json.dumps(labels), encoding="utf-8")
    for label, name in labels.items():
        print(f"\n===== {label} =====")
        for r in picks:
            print(r["dub"])
            display(Audio(str(next(out.glob(f"{name}_{r['id']}.*")))))
    print("\nNghe xong, chấm mỗi giọng 1–5 điểm rồi chạy ô tiếp theo để xem giọng nào là engine nào.")


def reveal():
    for label, name in RESULTS["tts"]["blind_key"].items():
        print(f"{label} = {name}: {RESULTS['tts']['stats'][name]}")


def save_report():
    dest = DRIVE / "bench"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã lưu {path}\nCopy nội dung dưới đây gửi cho Claude:\n")
    print(json.dumps(RESULTS, ensure_ascii=False))
