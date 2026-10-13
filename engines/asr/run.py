"""WhisperX: audio → asr.json (segments + timestamp từng từ). Chạy trong venv riêng engines/asr."""
import argparse
import gc
import json
import time

import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--lang", default=None)
ap.add_argument("--model", default="large-v3")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

t = {}
t0 = time.time()
model = whisperx.load_model(a.model, DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8", language=a.lang)
audio = whisperx.load_audio(a.audio)
t["load_s"] = round(time.time() - t0, 1)

t0 = time.time()
res = model.transcribe(audio, batch_size=a.batch, language=a.lang)
t["transcribe_s"] = round(time.time() - t0, 1)
lang = res["language"]
del model  # giải phóng VRAM trước khi nạp model căn chỉnh
gc.collect()

t0 = time.time()
for attempt in range(3):  # model căn chỉnh tải từ download.pytorch.org, hay bị ngắt kết nối giữa chừng
    try:
        align_model, meta = whisperx.load_align_model(language_code=lang, device=DEVICE)
        break
    except OSError:  # URLError, ConnectionResetError đều là OSError
        if attempt == 2:
            raise
        time.sleep(10)
res = whisperx.align(res["segments"], align_model, meta, audio, DEVICE, return_char_alignments=False)
t["align_s"] = round(time.time() - t0, 1)

segments = [
    {
        "start": s["start"],
        "end": s["end"],
        "text": s["text"].strip(),
        # số và ký hiệu đôi khi không căn chỉnh được nên thiếu s/e
        "words": [{"w": w["word"], "s": w.get("start"), "e": w.get("end")} for w in s.get("words", [])],
    }
    for s in res["segments"]
]
with open(a.out, "w", encoding="utf-8") as f:
    json.dump({"language": lang, "duration": len(audio) / 16000, "timing": t, "segments": segments},
              f, ensure_ascii=False, indent=1)
print(json.dumps({"language": lang, "segments": len(segments), "device": DEVICE, **t}))
