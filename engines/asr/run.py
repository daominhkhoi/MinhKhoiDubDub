"""WhisperX: audio → asr.json (segments + timestamp từng từ). Chạy trong venv riêng engines/asr."""
import argparse
import gc
import json
import time

import whisperx

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--lang", default=None)
ap.add_argument("--model", default="large-v3")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

t = {}
t0 = time.time()
model = whisperx.load_model(a.model, "cuda", compute_type="float16", language=a.lang)
audio = whisperx.load_audio(a.audio)
t["load_s"] = round(time.time() - t0, 1)

t0 = time.time()
res = model.transcribe(audio, batch_size=a.batch, language=a.lang)
t["transcribe_s"] = round(time.time() - t0, 1)
lang = res["language"]
del model  # giải phóng VRAM trước khi nạp model căn chỉnh
gc.collect()

t0 = time.time()
align_model, meta = whisperx.load_align_model(language_code=lang, device="cuda")
res = whisperx.align(res["segments"], align_model, meta, audio, "cuda", return_char_alignments=False)
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
print(json.dumps({"language": lang, "segments": len(segments), **t}))
