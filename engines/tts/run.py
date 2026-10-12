"""viXTTS: items.json [{id, text, out}] + giọng mẫu → wav 24 kHz. Chạy trong venv riêng engines/tts."""
import argparse
import json
import time

import soundfile as sf
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts import tokenizer as xtok
from TTS.tts.models.xtts import Xtts

# shortcut: coqui-tts gốc chưa nhận "vi"; viXTTS chỉ cần chữ thường + gọn khoảng trắng.
# Chuyển sang fork tiếng Việt chính thức nếu bài nghe cho thấy đọc sai.
_orig_pre = xtok.VoiceBpeTokenizer.preprocess_text


def _pre(self, txt, lang):
    return " ".join(txt.lower().split()) if lang == "vi" else _orig_pre(self, txt, lang)


xtok.VoiceBpeTokenizer.preprocess_text = _pre

ap = argparse.ArgumentParser()
ap.add_argument("items")
ap.add_argument("ref", help="wav giọng mẫu 6–12 giây")
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

t0 = time.time()
ckpt = snapshot_download("capleaf/viXTTS")
cfg = XttsConfig()
cfg.load_json(f"{ckpt}/config.json")
model = Xtts.init_from_config(cfg)
model.load_checkpoint(cfg, checkpoint_dir=ckpt, use_deepspeed=False)
model.cuda()
latent, emb = model.get_conditioning_latents(
    audio_path=[a.ref], gpt_cond_len=cfg.gpt_cond_len,
    max_ref_length=cfg.max_ref_len, sound_norm_refs=cfg.sound_norm_refs,
)
load_s = round(time.time() - t0, 1)

rows = []
with open(a.items, encoding="utf-8") as f:
    items = json.load(f)
for it in items:
    t0 = time.time()
    out = model.inference(
        text=it["text"], language="vi", gpt_cond_latent=latent, speaker_embedding=emb,
        temperature=0.3, repetition_penalty=10.0, top_k=30, top_p=0.85,
        speed=a.speed, enable_text_splitting=True,
    )
    sf.write(it["out"], out["wav"], 24000)
    rows.append({"id": it["id"], "dur": round(len(out["wav"]) / 24000, 2), "synth_s": round(time.time() - t0, 2)})
print(json.dumps({"load_s": load_s, "items": rows}))
