# SuperDubDub: Kế hoạch triển khai (bản miễn phí 100%, chạy local)

> Phiên bản: 1.0, ngày 12/10/2026
> Trạng thái: **Bản nháp chờ duyệt.** Chưa viết dòng code nào.

---

## 0. Tóm tắt

| Hạng mục | Quyết định |
|---|---|
| Người dùng | 1 người (cá nhân), không thương mại |
| Nơi chạy | Laptop: Ryzen 5 5625U (6C/12T), RAM 15.4 GB, **không có GPU NVIDIA**, đĩa D còn 50 GB |
| Ngôn ngữ nguồn | Tiếng Anh, tiếng Nhật, tiếng Trung giản thể |
| Độ dài video | 10–60 phút |
| Chi phí | **0 đồng.** Không dùng API trả phí, không cần thẻ tín dụng |
| Đầu ra | MP4 có track tiếng Việt (thuyết minh hoặc lồng tiếng), phụ đề SRT, và project có thể chỉnh sửa |
| Tốc độ (ước lượng, sẽ đo ở Phase 0) | Video 30 phút mất khoảng **1.5–2.5 giờ** xử lý trên CPU. Phù hợp chạy qua đêm |
| Thời gian làm | Khoảng **7–10 tuần** nếu làm bán thời gian |

**Ba nguyên tắc xuyên suốt:**

1. **Không bao giờ nạp nhiều model nặng cùng lúc.** RAM 15 GB chỉ đủ cho một model tại một thời điểm. Mỗi bước chạy trong một tiến trình con riêng (subprocess), tiến trình kết thúc thì RAM được giải phóng.
2. **`segments.json` là nguồn dữ liệu duy nhất.** Mọi bước đọc và ghi file này. Sửa một câu thì chỉ tạo lại audio của câu đó, không chạy lại cả video.
3. **Mỗi bước đều có thể chạy lại (resume).** Nếu file kết quả của một bước đã tồn tại thì bỏ qua bước đó. Tắt máy giữa chừng, lần sau chạy lại sẽ tiếp tục từ chỗ dừng.

---

## 1. Ràng buộc và hệ quả

| Ràng buộc | Hệ quả thiết kế |
|---|---|
| Không có CUDA | Dùng model chạy CPU: faster-whisper int8, Ollama trên CPU, TTS có hỗ trợ CPU. Sẽ thử tăng tốc bằng Vulkan/DirectML trên iGPU AMD ở Phase 0, nhưng không phụ thuộc vào nó |
| RAM 15 GB | LLM tối đa khoảng 12B tham số (bản lượng tử hoá Q4, ~8 GB). Audio 60 phút phải xử lý theo đoạn khoảng 10 phút |
| Python 3.14 đã cài | **Quá mới** với hệ sinh thái torch. Dùng `uv` cài Python 3.11 riêng cho dự án, không đụng vào bản 3.14 đang có |
| Các thư viện ML xung đột phiên bản torch | **Mỗi engine một venv riêng** (ASR, TTS, tách nguồn), gọi qua subprocess bằng file JSON. App chính chỉ dùng thư viện nhẹ |
| Ổ C còn 57 GB | Model Hugging Face và job đặt ở **D:** (cấu hình `HF_HOME`, `JOBS_DIR` trong config dự án). Model Ollama (~13 GB) vẫn ở C: để không làm mất các model hiện có |
| Miễn phí, dùng cá nhân | Được dùng model có license phi thương mại (CC-BY-NC, Coqui CPML). **Không phân phối** video đã lồng tiếng của nội dung có bản quyền |

---

## 2. Kiến trúc

### 2.1 Luồng xử lý

```
input.mp4
  │
  ├─[1 ingest]     ffprobe kiểm tra file → audio16k.wav (cho ASR) + audio48.wav (cho mix)
  ├─[2 separate]*  Demucs/MDX → vocals.wav + mne.wav          (* chỉ cần ở chế độ lồng tiếng
  │                                                              hoặc nguồn có nhạc nền dày)
  ├─[3 asr]        WhisperX: VAD → nhận dạng → căn chỉnh từng từ → asr.json
  ├─[3b diarize]*  pyannote → gán speaker cho từng segment       (* tuỳ chọn: phim nhiều nhân vật)
  ├─[4 segment]    chia thành câu/cue theo luật phụ đề → segments.json
  ├─[5 context]    LLM pass 1: tóm tắt, nhân vật, bảng xưng hô, glossary → context.json
  ├─[6 translate]  LLM pass 2: điền sub + dub cho từng segment → segments.json
  │      ⏸ dừng ở đây nếu bật "duyệt trước khi lồng tiếng" (sửa trong web UI)
  ├─[7 tts]        tổng hợp giọng từng segment → tts/<id>_<hash>.wav
  ├─[8 fit]        khớp thời lượng: tăng tốc / mượn khoảng lặng / LLM rút gọn câu → TTS lại
  ├─[9 mix]        ducking + loudnorm → vi_mix.wav
  └─[10 mux]       ffmpeg -c:v copy (không encode lại video) → out.mp4 + vi.srt
```

### 2.2 Hai chế độ đầu ra

| Chế độ | Cách làm | Cần tách nguồn? | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| **Thuyết minh** (mặc định) | Giữ nguyên audio gốc, giảm 12–18 dB khi có giọng Việt | **Không** | Nhanh, không có lỗi do tách nguồn, không cần khớp khẩu hình | Vẫn nghe thấy giọng gốc nhỏ phía sau |
| Lồng tiếng | Bỏ giọng gốc, giữ nhạc nền và tiếng động (M&E) | Có | Giống bản lồng tiếng thật | Chậm hơn khoảng 15–30 phút cho mỗi video, có thể nghe thấy giọng gốc rò rỉ |

### 2.3 Hệ thống (một máy, một người dùng)

```
Trình duyệt (localhost:8000)
     │  HTML + JS thuần, không cần bước build
     ▼
FastAPI  ── SQLite (jobs: id, status, step, progress, error)
     │
     ▼
1 tiến trình worker (vòng lặp đọc job pending từ SQLite, mỗi lần chạy 1 job)
     │  gọi tuần tự từng bước
     ├── subprocess → engines/asr/.venv   (WhisperX)
     ├── subprocess → engines/sep/.venv   (Demucs)
     ├── subprocess → engines/tts/.venv   (viXTTS / engine được chọn)
     ├── HTTP → Ollama localhost:11434    (LLM, API chuẩn OpenAI)
     └── ffmpeg                           (cắt audio, mix, mux)
```

**Những thứ cố ý không dùng:**
- Redis, Celery, Postgres, Docker, S3: một người dùng trên một máy thì SQLite và thư mục trên đĩa là đủ.
- React và các công cụ build: giao diện chỉ cần một bảng segment và một trình phát video.
- Nâng cấp sau nếu thật sự cần.

### 2.4 Cấu trúc thư mục

```
superdubdub/
├── PLAN.md
├── pyproject.toml            # venv chính: fastapi, uvicorn, numpy, soundfile, httpx, pysubs2
├── config.toml               # đường dẫn, model mặc định, endpoint LLM, giọng đọc
├── dub/
│   ├── cli.py                # dub run <video> --src ja --mode voiceover
│   ├── pipeline.py           # thứ tự các bước + logic resume
│   ├── steps/                # mỗi bước trong §2.1 là một file
│   │   ├── ingest.py  separate.py  asr.py  segment.py  translate.py
│   │   └── tts.py  fit.py  mix.py  mux.py
│   ├── llm.py                # client chuẩn OpenAI (Ollama mặc định; Gemini free tier tuỳ chọn)
│   ├── names.py              # tên riêng: tiếng Trung → Hán-Việt, tiếng Nhật → Romaji
│   └── web/                  # app.py + static/index.html, editor.html
├── engines/                  # mỗi engine một venv + run.py (đọc JSON vào, ghi JSON/WAV ra)
│   ├── asr/  sep/  tts/
├── tests/                    # fit solver, đếm âm tiết, chia cue, schema segments
└── D:\superdubdub-data\      # (ngoài repo) models/, jobs/<id>/
```

### 2.5 Hợp đồng dữ liệu: `segments.json`

```json
{
  "job": {"src_lang": "ja", "mode": "voiceover", "duration": 1843.2},
  "speakers": {"S1": {"name": "Tanjiro", "gender": "m", "voice": "male_young"}},
  "segments": [
    {
      "id": 47, "speaker": "S1",
      "src_start": 312.48, "src_end": 315.90,
      "src_text": "絶対に諦めない！",
      "words": [{"w": "絶対", "s": 312.48, "e": 312.95}],
      "sub": "Tớ tuyệt đối không bỏ cuộc!",
      "dub": "Tớ tuyệt đối không bỏ cuộc!",
      "dub_syllables": 6,
      "emotion": "determined",
      "tts": {"file": "tts/47_a1b2c3.wav", "dur": 2.71},
      "placed_start": 312.48, "rate": 1.0,
      "flags": []
    }
  ]
}
```

`flags` chứa các cảnh báo như `"needs_review"` (đã tăng tốc tối đa mà vẫn không vừa) hoặc `"asr_low_conf"`. Web UI hiển thị các segment có cờ lên đầu danh sách.

### 2.6 Bố cục thư mục của một job

```
jobs/<id>/
  input.mp4  audio16k.wav  audio48.wav
  stems/vocals.wav  stems/mne.wav          # chỉ khi có tách nguồn
  asr.json  context.json  segments.json
  tts/<seg_id>_<hash(dub+voice+rate)>.wav  # cache theo nội dung
  vi_mix.wav  out.mp4  vi.srt  log.txt
```

Một video 60 phút chiếm khoảng 3–4 GB. Sau khi xuất xong, có nút "Dọn file tạm" để xoá stem và wav trung gian.

---

## 3. Lựa chọn công nghệ (tất cả miễn phí)

Các mục đánh dấu ⚖ sẽ được chốt bằng số liệu đo ở Phase 0, không chọn theo cảm tính.

### 3.1 Nhận dạng giọng nói (ASR)

| Ngôn ngữ | Mặc định | Dự phòng | Ghi chú |
|---|---|---|---|
| Anh | WhisperX + `large-v3-turbo` int8 | `distil-large-v3` (nhanh hơn) | Căn chỉnh từng từ bằng wav2vec2 có sẵn trong WhisperX |
| Nhật | WhisperX + `large-v3-turbo` int8 | `large-v3` int8 (chính xác hơn, chậm gấp ~3 lần) | ⚖ So sánh CER giữa turbo và large-v3 trên clip anime |
| Trung | WhisperX + `large-v3-turbo` int8 | **FunASR Paraformer-zh** (có timestamp từng chữ và dấu câu, chạy CPU nhanh) | ⚖ Whisper đôi khi trả về chữ phồn thể hoặc thiếu dấu câu. Dùng `initial_prompt` giản thể, hoặc chuyển đổi bằng OpenCC |

**Chống "bịa chữ" (hallucination):** bật lọc VAD, đặt `condition_on_previous_text=False`, bỏ segment có `compression_ratio > 2.4` hoặc `avg_logprob < -1.0` và gắn cờ cho nó.

**Tăng tốc tuỳ chọn:** thử whisper.cpp với backend Vulkan trên iGPU Radeon. Nếu nhanh hơn trên 2 lần thì chuyển sang dùng.

**Diarization (pyannote 3.1):** miễn phí nhưng cần token Hugging Face và đồng ý điều khoản model. Trên CPU khá chậm, nên **mặc định tắt**. Chỉ bật cho phim nhiều nhân vật.

### 3.2 Dịch thuật (LLM local qua Ollama)

| Ứng viên | RAM (Q4) | Tốc độ CPU (ước lượng) | Ghi chú |
|---|---|---|---|
| `qwen2.5:7b` (**đã có trên máy**) | ~5 GB | ~6–8 token/giây | Baseline |
| `qwen3:8b` | ~5.5 GB | ~5–7 token/giây | Mạnh với tiếng Trung và tiếng Nhật |
| `gemma3:12b` | ~8 GB | ~3–4 token/giây | Tiếng Việt thường tự nhiên hơn, nhưng chậm |

⚖ Chọn bằng bài test mù trên 50 câu × 3 ngôn ngữ, chấm theo: đúng nghĩa, xưng hô, độ tự nhiên. Nếu Ollama có bản mới hơn cùng cỡ thì đưa vào cùng bài test.

**Tuỳ chọn nhanh và tốt hơn (vẫn miễn phí):** dùng free tier của Gemini API qua endpoint tương thích OpenAI. Chỉ cần đổi `base_url` và `model` trong `config.toml`, không phải sửa code. Đổi lại, free tier giới hạn số request, cần Internet, và **dữ liệu có thể được Google dùng để cải thiện sản phẩm**. Mặc định vẫn là Ollama.

**Kỹ thuật dịch:**

- **Pass 1** (một lần cho mỗi video): tóm tắt nội dung, xác định thể loại và văn phong, danh sách nhân vật, **bảng xưng hô theo từng cặp nhân vật** (anh/em, cậu/tớ, ta/ngươi…), glossary thuật ngữ.
- **Pass 2**: mỗi chunk 15–25 segment, kèm 5 câu đã dịch trước đó làm ngữ cảnh. Thiết lập:
  - `format` = JSON schema (structured outputs của Ollama).
  - `temperature 0.3`, `num_ctx 8192`.
  - Kiểm tra số lượng và ID segment khớp 1:1. Sai thì thử lại tối đa 2 lần, sau đó gắn cờ.
- **Mỗi segment có 2 bản dịch:**
  - `sub`: tối đa 2 dòng × 42 ký tự, tốc độ đọc ≤ ~17 ký tự/giây.
  - `dub`: tối đa `syllable_budget` âm tiết, viết sẵn ở **dạng đọc** (số, ngày, đơn vị ghi bằng chữ).
- **Đếm âm tiết tiếng Việt bằng `len(text.split())`.** Tiếng Việt là ngôn ngữ đơn âm tiết nên cách đếm này chính xác.
  - `syllable_budget = floor((src_end - src_start + khoảng lặng phía sau) × voice_rate)`
  - `voice_rate` đo riêng cho từng giọng TTS ở Phase 0, thường khoảng 4.5–5.5 âm tiết/giây.
- **Tên riêng không giao cho LLM tự xử lý** (model 7B làm phần này kém):
  - Tiếng Trung: tra bảng phiên âm Hán-Việt theo từng chữ (dữ liệu cộng đồng, dạng `ChinesePhienAmWords`) → 张伟 thành Trương Vĩ.
  - Tiếng Nhật: giữ Romaji theo thông lệ cộng đồng anime (`pykakasi`/`cutlet`) → 竈門炭治郎 thành Kamado Tanjiro.
  - Kết quả đưa vào glossary của pass 1. Người dùng sửa được, và glossary được **lưu theo series** để dùng lại cho các tập sau.

### 3.3 Tổng hợp giọng nói tiếng Việt (TTS)

Vì dự án dùng cá nhân, được phép dùng model có license phi thương mại.

| Ứng viên | Clone giọng | Chạy CPU | Chất lượng (kỳ vọng) | Rủi ro |
|---|---|---|---|---|
| **viXTTS** (XTTS-v2 fine-tune tiếng Việt) | Có (mẫu 6–10 giây) | Được, chậm (RTF ~1–3) | Cao, có cảm xúc | License CPML: chỉ phi thương mại (phù hợp). Đôi khi lặp hoặc nuốt chữ ở câu rất ngắn |
| Checkpoint **F5-TTS tiếng Việt** của cộng đồng | Có | Được, chậm (phụ thuộc số bước NFE) | Cao | Trọng số CC-BY-NC (phù hợp). Chất lượng tuỳ checkpoint |
| **VieNeu-TTS** (nếu còn được phát hành) | Có | Nhắm tới CPU | Cần kiểm chứng | Phải kiểm tra license và tình trạng repo |
| **edge-tts** (giọng HoaiMy, NamMinh) | Không | Không tốn CPU (chạy trên server Microsoft) | Tốt, ổn định | **Endpoint không chính thức**, có thể ngừng bất cứ lúc nào, cần Internet. Chỉ dùng làm chế độ "xem trước nhanh" |
| **Piper** vi_VN | Không | Rất nhanh | Giọng máy | Dự phòng cuối cùng khi offline |

⚖ **Bài nghe mù (MOS):** 20 câu × 5 engine, bạn chấm 1–5 điểm mà không biết engine nào. Mặc định tạm thời là **viXTTS** (offline, clone giọng, tự nhiên). Code chỉ cần **một hàm** `synthesize(text, voice, speed) -> wav`. Ban đầu chỉ có 1 engine, thêm engine thứ hai khi bài test chứng minh cần.

**Thư viện giọng:** đặt các file mẫu 6–10 giây (giọng nam/nữ, trẻ/già) trong `voices/`. Khi bật diarization, có thể **clone giọng nhân vật từ chính video** bằng cách lấy đoạn thoại sạch nhất của speaker đó.

### 3.4 Tách nguồn, mix, mux

| Hạng mục | Công cụ | Ghi chú |
|---|---|---|
| Tách nguồn | ⚖ Demucs `htdemucs` và MDX-Net ONNX (qua `audio-separator`) | Xử lý theo đoạn 10 phút, overlap 5 giây để không tràn RAM. Thử DirectML trên iGPU nếu thư viện hỗ trợ |
| Ducking | numpy: tính envelope từ timeline đã biết trước, không dùng compressor | Giảm nền 8–12 dB (lồng tiếng) hoặc 12–18 dB (thuyết minh). Fade vào 150 ms, fade ra 250 ms |
| Co giãn thời gian | Ưu tiên tham số `speed` của TTS (tạo lại câu). Dự phòng: ffmpeg `rubberband` (có trong bản build full của Gyan), hoặc `atempo` | Không nghe ra khi ≤ 1.1x |
| Chuẩn hoá âm lượng | ffmpeg `loudnorm` 2 lượt: `I=-16:TP=-1.5:LRA=11` | |
| Mux | `ffmpeg -c:v copy`. Track 1: tiếng Việt, track 2: tiếng gốc, phụ đề `mov_text` | Không encode lại video nên chỉ mất vài giây |
| Burn-in phụ đề | libx264 `-preset veryfast` (tuỳ chọn) | **Chậm trên CPU** (~0.5–1x realtime). Chỉ làm khi người dùng yêu cầu |
| Phụ đề | `pysubs2` → SRT/ASS | |

---

## 4. Thuật toán khó

### 4.1 Khớp thời lượng (bước 8)

Với mỗi segment: `W` là cửa sổ khả dụng (tính đến lúc câu tiếp theo bắt đầu, trừ 100 ms), `D` là thời lượng TTS sau khi cắt khoảng lặng ở đầu và cuối, `r = D / W`.

| r | Hành động |
|---|---|
| ≤ 1.0 | Đặt tại thời điểm bắt đầu của từ đầu tiên. Phần thừa để im lặng (không kéo chậm giọng) |
| 1.0–1.15 | Tạo lại TTS với `speed = r` |
| 1.15–1.3 | Mượn khoảng lặng: đặt sớm hơn tối đa 200 ms, kéo dài sang gap phía sau, kết hợp speed ≤ 1.2 |
| > 1.3 | LLM rút gọn `dub` theo ngân sách âm tiết mới, rồi TTS lại. Tối đa 2 vòng |
| Vẫn không vừa | Ép tối đa 1.35x và gắn cờ `needs_review` |

Thuật toán là greedy từ trái sang phải. Biến `t_free` lưu thời điểm sớm nhất được phép đặt câu tiếp theo, đảm bảo **không bao giờ có hai câu chồng lên nhau**. Thuật toán viết dưới dạng hàm thuần (không I/O) và có unit test.

### 4.2 Luật chia cue phụ đề (bước 4)

- Cắt câu theo dấu câu và khoảng lặng giữa các từ > 0.5 giây. Mỗi cue dài 0.83–7 giây, cách nhau ≥ 2 frame.
- Tối đa 2 dòng × 42 ký tự. Ngắt dòng ở dấu phẩy hoặc ranh giới cụm từ, không ngắt giữa một từ ghép.
- **Có 2 track phụ đề:**
  - `vi.srt` (theo giọng lồng tiếng): timing lấy chính xác từ `placed_start` của TTS.
  - `vi_orig.srt` (theo giọng gốc): timing lấy từ timestamp từng từ của ASR.

### 4.3 Quản lý RAM

```
Bước       Đỉnh RAM (ước lượng)   Cách kiểm soát
asr        ~3 GB                  int8, batch_size nhỏ, tiến trình con tự thoát
separate   ~4 GB                  chia đoạn 10 phút
translate  5–9 GB                 dừng Ollama model sau bước này (keep_alive=0)
tts        ~3 GB                  tiến trình con, xử lý từng segment, cache
mix        ~2 GB                  float32 mono 48k cho 60 phút ≈ 0.7 GB mỗi track
```

---

## 5. Lộ trình triển khai

### Phase 0: Môi trường và benchmark (1 tuần)

**Việc cần làm:**
1. `git init`, tạo `.gitignore` (bỏ qua `jobs/`, `models/`, `.venv/`).
2. Cài đặt (miễn phí): `winget install Gyan.FFmpeg`, `winget install astral-sh.uv`, `uv python install 3.11`.
3. Đặt `HF_HOME=D:\superdubdub-data\models` **trong config của dự án** (không đặt biến môi trường toàn hệ thống, để không ảnh hưởng các dự án khác). Ollama giữ thư mục mặc định: nếu đổi `OLLAMA_MODELS` thì Ollama sẽ không thấy các model `smartparking-qwen*` hiện có.
4. **Bộ test** (đặt trong `videoDemo/`, git đã bỏ qua thư mục này): 6 clip × 3 phút gồm EN (talk), EN (phim), JA (anime), JA (vlog), ZH (drama), ZH (show). Gõ tay transcript chuẩn cho 1 phút mỗi clip để tính WER/CER.
5. Chạy benchmark ASR, LLM, TTS (bài nghe mù), tách nguồn. Đo thời gian xử lý so với độ dài clip (RTF) và RAM đỉnh.
6. Ghi kết quả vào `BENCHMARK.md` và chốt các mục ⚖.

**Tiêu chí hoàn thành:**
- Mọi engine ứng viên chạy được trên laptop.
- Có bảng số liệu.
- Đã chốt engine mặc định cho ASR, LLM, TTS.
- Đã đo `voice_rate` của giọng mặc định.

### Phase 1: Pipeline CLI (2–3 tuần)

**Việc cần làm:**
- Viết các bước 1, 3, 4, 5, 6, 7, 9, 10 theo §2.1 ở chế độ thuyết minh. Chưa có tách nguồn, chưa có fit nâng cao (TTS quá dài thì tạm tăng tốc đơn giản).
- `dub run video.mp4 --src ja`, logic resume, `log.txt`, hiển thị tiến độ trên terminal.
- Unit test: đếm âm tiết, kiểm tra schema segments, kiểm tra ID khi dịch.

**Tiêu chí hoàn thành:**
- Clip 10 phút chạy từ đầu đến cuối mà không cần can thiệp.
- Kill tiến trình giữa chừng rồi chạy lại thì tiếp tục đúng bước.
- Mở được `out.mp4` và đổi được giữa 2 track audio.

### Phase 2: Đồng bộ và chất lượng âm thanh (1–2 tuần)

**Việc cần làm:**
- Fit solver §4.1 và vòng LLM rút gọn câu.
- Luật cue §4.2, ducking envelope, loudnorm 2 lượt.
- Bước 2 (tách nguồn) và chế độ lồng tiếng.
- Unit test cho fit solver: không chồng câu, tôn trọng giới hạn tốc độ, gắn cờ đúng.

**Tiêu chí hoàn thành** (đo trên bộ test):
- Dưới 5% segment có `rate > 1.15`.
- Không có câu nào chồng lên nhau.
- Loudness đạt −16 ± 1 LUFS.
- Không còn cue phụ đề vi phạm luật.

### Phase 3: Giao diện web local (2–3 tuần)

**Việc cần làm:**
- FastAPI, SQLite, worker. Trang **Danh sách job** gồm: upload, chọn ngôn ngữ và chế độ, thanh tiến độ (SSE).
- Trang **Editor**:
  - Video player cạnh bảng segment. Bấm vào dòng nào thì video tua đến đó.
  - Sửa `sub`, `dub`, speaker, giọng. Nút "Nghe thử câu này".
  - Lọc các segment có cờ. Sửa glossary và bảng xưng hô.
- Nút **Tạo lại**: chỉ TTS lại các segment đã sửa, rồi fit, mix và mux lại.
- Tuỳ chọn "Dừng để duyệt bản dịch trước khi lồng tiếng".

**Tiêu chí hoàn thành:**
- Sửa 1 câu và xuất lại video trong dưới 1 phút.
- Toàn bộ thao tác thực hiện được trên trình duyệt, không cần dùng terminal.

### Phase 4: Nâng cao (tuỳ chọn, làm theo nhu cầu thực tế)

- Diarization và clone giọng từng nhân vật từ chính video.
- Bộ nhớ theo series: glossary và bảng xưng hô dùng lại giữa các tập.
- Hàng đợi chạy qua đêm cho nhiều video.
- Burn-in phụ đề.
- Engine TTS thứ hai, nếu bài nghe mù cho thấy đáng làm.

**Tổng thời gian:** khoảng 7–10 tuần bán thời gian. Mỗi phase đều ra được sản phẩm dùng được ngay.

---

## 6. Ước lượng hiệu năng (sẽ thay bằng số đo thật ở Phase 0)

Video 30 phút, chế độ thuyết minh, chạy trên CPU Ryzen 5 5625U:

| Bước | Thời gian |
|---|---|
| Ingest | < 1 phút |
| ASR (turbo int8) + căn chỉnh từng từ | 12–20 phút |
| Dịch (2 pass, model 7–8B) | 30–50 phút |
| TTS (~20 phút giọng nói, RTF ~1.5) | 25–40 phút |
| Fit + mix + mux | 3–5 phút |
| **Tổng** | **~1.2–2 giờ** (cộng 10–20 phút nếu tách nguồn, 10–20 phút nếu diarization) |

**Mẹo vận hành:** cắm sạc, bật chế độ Power "Best performance", để máy chạy qua đêm.

---

## 7. Rủi ro và cách giảm thiểu

| Rủi ro | Khả năng | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| LLM 7–12B dịch kém tự nhiên, sai xưng hô | Cao | Cao | Dịch 2 pass, bảng xưng hô, tên riêng xử lý bằng code. Editor để sửa tay. Có lựa chọn chuyển sang Gemini free tier |
| Chạy CPU quá chậm với video 60 phút | Trung bình | Trung bình | Chạy qua đêm, resume, cache. Thử Vulkan/DirectML ở Phase 0 |
| Xung đột phiên bản thư viện Python/torch | Cao | Trung bình | Mỗi engine một venv, Python 3.11, khoá phiên bản bằng `uv.lock` |
| Tràn RAM (OOM) | Trung bình | Cao | Mỗi bước một tiến trình con, chia đoạn audio, `keep_alive=0` cho Ollama |
| Whisper bịa chữ ở đoạn nhạc | Trung bình | Trung bình | VAD, ngưỡng loại segment, gắn cờ. Bật tách nguồn trước ASR cho anime/MV |
| TTS lặp hoặc nuốt chữ ở câu rất ngắn | Trung bình | Thấp | Gộp câu ngắn cùng speaker, kiểm tra thời lượng bất thường và tự TTS lại |
| edge-tts ngừng hoạt động | Trung bình | Thấp | Chỉ dùng cho xem trước nhanh, engine chính chạy offline |
| Repo hoặc trọng số model cộng đồng bị gỡ | Thấp | Trung bình | Lưu bản sao trọng số đã chọn trên ổ D |
| Bản quyền video nguồn | n/a | Pháp lý | Chỉ dùng cá nhân, không đăng lại nội dung đã lồng tiếng |

---

## 8. Chỉ số theo dõi chất lượng

- **ASR:** WER (tiếng Anh) và CER (tiếng Nhật, tiếng Trung) trên bộ test.
- **Dịch:** điểm bài test mù và **tỷ lệ câu phải sửa tay** trong editor (chỉ số quan trọng nhất).
- **Đồng bộ:** phân bố `rate`, % segment có cờ `needs_review`, số câu chồng nhau (phải bằng 0).
- **Âm thanh:** LUFS tích hợp, true peak.
- **Hiệu năng:** RTF của từng bước, RAM đỉnh.

---

## 9. Quyết định cần bạn duyệt trước khi bắt đầu Phase 0

1. Đồng ý cài **ffmpeg** và **uv** qua winget, và Python 3.11 qua uv (không đụng bản 3.14 đang có).
2. Đồng ý đặt dữ liệu và model tại `D:\superdubdub-data` (cần khoảng 20–25 GB).
3. Đồng ý tải model LLM thứ hai và thứ ba để so sánh (khoảng 5–8 GB mỗi model). Có thể xoá sau khi chốt.
4. Có muốn thử **Gemini free tier** trong bài benchmark dịch không? (Cần tạo API key miễn phí. Dữ liệu có thể được Google sử dụng.)
5. Chuẩn bị 6 clip test theo §5 Phase 0, hoặc để tôi gợi ý cách lấy clip mẫu hợp lệ.
