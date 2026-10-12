# SuperDubDub: Kế hoạch triển khai (bản miễn phí 100%, chạy trên Google Colab)

> Phiên bản: 2.0, ngày 12/10/2026
> Thay thế bản 1.0 (chạy local trên CPU). Lý do: chạy model local chiếm 5–9 GB RAM và mất 1.5–2 giờ cho mỗi video trên laptop.
> Trạng thái: **Bản nháp chờ duyệt.** Chưa viết dòng code nào.

---

## 0. Tóm tắt

| Hạng mục | Quyết định |
|---|---|
| Người dùng | 1 người (cá nhân), không thương mại |
| Nơi chạy AI | **Google Colab miễn phí** (GPU T4 16 GB VRAM, ~12.7 GB RAM). Laptop chỉ mở trình duyệt |
| Lưu trữ | **Google Drive** (`MyDrive/SuperDubDub/`): video đầu vào, tiến độ job, kết quả |
| Code | GitHub public `daominhkhoi/MinhKhoiDubDub`. Notebook tự clone code mỗi phiên |
| Ngôn ngữ nguồn | Tiếng Anh, tiếng Nhật, tiếng Trung giản thể |
| Độ dài video | 10–60 phút |
| Chi phí | **0 đồng**. Không cần thẻ tín dụng |
| Đầu ra | MP4 có track tiếng Việt (thuyết minh hoặc lồng tiếng), phụ đề SRT, project chỉnh sửa được |
| Tốc độ (ước lượng, đo ở Phase 0) | Video 30 phút mất khoảng **15–25 phút**, cộng **5–7 phút cài đặt** cho mỗi phiên Colab |
| Thời gian làm | Khoảng **6–8 tuần** nếu làm bán thời gian |

**Bốn nguyên tắc xuyên suốt:**

1. **Notebook là giao diện duy nhất.** Editor hiển thị ngay trong ô output của notebook, không mở link public ra ngoài (xem §1, luật của Colab).
2. **Chấp nhận việc Colab có thể ngắt bất cứ lúc nào.** Mỗi bước xong là lưu kết quả lên Drive. Chạy lại thì tiếp tục từ bước còn dở.
3. **`segments.json` là nguồn dữ liệu duy nhất.** Sửa một câu thì chỉ tạo lại audio của câu đó, không chạy lại cả video.
4. **Mỗi engine AI chạy trong tiến trình con riêng.** Tiến trình kết thúc thì VRAM và RAM được giải phóng, nhờ vậy các model không phải nằm cùng lúc trong bộ nhớ.

---

## 1. Ràng buộc của Colab miễn phí và cách xử lý

| Ràng buộc | Cách xử lý |
|---|---|
| **Free tier cấm** dùng notebook làm web server rồi truy cập chủ yếu qua giao diện web bên ngoài (ngrok, cloudflared, link share công khai) | Editor hiển thị **trong notebook** bằng `google.colab.output.serve_kernel_port_as_iframe`. Không tạo tunnel hay link public |
| Ngắt sau khoảng 90 phút không tương tác, phiên tối đa khoảng 12 giờ, có lúc không cấp GPU | Lưu tiến độ lên Drive sau mỗi bước, có resume. Khi job đang chạy thì để tab mở. Nếu không có GPU: các bước nhẹ vẫn chạy được, các bước cần GPU thì chờ |
| Mỗi phiên là một máy trắng | Ô cài đặt dùng `uv` (rất nhanh) và khoá phiên bản. Model tải lại mỗi phiên (băng thông của Colab nhanh). Nếu tải quá chậm thì mới cache model trên Drive |
| Ghi file lớn trực tiếp lên Drive rất chậm | Xử lý trên ổ `/content` (SSD của máy Colab). Hết mỗi bước mới **chép kết quả lên Drive** |
| Drive miễn phí 15 GB | Tự xoá file trung gian (stem, wav) sau khi xuất video. Chỉ giữ `segments.json`, file TTS đã cache, video kết quả |
| Ảnh môi trường của Colab (Python, torch) thay đổi theo thời gian | Mỗi engine một venv riêng tạo bằng `uv`, khoá phiên bản bằng file lock. Không phụ thuộc torch có sẵn của Colab |
| API key | Lưu trong **Colab Secrets** (biểu tượng 🔑): `GEMINI_API_KEY`, tuỳ chọn thêm `HF_TOKEN`. Đọc bằng `google.colab.userdata.get()`, không ghi vào code |

---

## 2. Kiến trúc

### 2.1 Tổng thể

```
Laptop: chỉ mở trình duyệt
   │  bấm "Open in Colab" từ README trên GitHub
   ▼
Colab notebook (GPU T4)
   ├─ Ô 1  Cài đặt   : mount Drive, git clone (chọn branch), uv tạo venv cho từng engine
   ├─ Ô 2  Chạy job   : chọn video trong Drive/input, ngôn ngữ, chế độ → pipeline (có resume)
   ├─ Ô 3  Editor     : FastAPI chạy nền, hiển thị bằng iframe ngay trong notebook
   └─ Ô 4  Xuất / dọn : mux video, chép kết quả về Drive, xoá file tạm
            │
            ├── subprocess → venv asr  (WhisperX, GPU)
            ├── subprocess → venv sep  (Demucs / RoFormer, GPU)
            ├── subprocess → venv tts  (viXTTS, GPU)
            ├── HTTPS     → Gemini API (dịch)
            └── ffmpeg    (có sẵn trên Colab)

Google Drive: MyDrive/SuperDubDub/
   ├── input/          ← bạn upload video vào đây
   ├── jobs/<id>/      ← tiến độ và kết quả (bản sao bền vững)
   └── series/<tên>/   ← glossary và bảng xưng hô dùng lại giữa các tập
```

### 2.2 Luồng xử lý

```
[1 ingest]     ffprobe kiểm tra file → audio16k.wav (ASR) + audio48.wav (mix)
[2 separate]*  Demucs/RoFormer → vocals.wav + mne.wav       (* chế độ lồng tiếng, hoặc nguồn có nhạc nền dày)
[3 asr]        WhisperX large-v3: VAD → nhận dạng → căn chỉnh từng từ → asr.json
[3b diarize]*  pyannote → gán speaker                       (* cần HF_TOKEN; bật cho phim nhiều nhân vật)
[4 segment]    chia câu/cue theo luật phụ đề → segments.json
[5 context]    Gemini pass 1: tóm tắt, nhân vật, bảng xưng hô, glossary → context.json
[6 translate]  Gemini pass 2: điền sub + dub cho từng segment
      ⏸ tuỳ chọn: dừng để duyệt bản dịch trong editor
[7 tts]        viXTTS (GPU) từng segment → tts/<id>_<hash>.wav
[8 fit]        khớp thời lượng: tăng tốc / mượn khoảng lặng / Gemini rút gọn câu → TTS lại
[9 mix]        ducking + loudnorm → vi_mix.wav
[10 mux]       ffmpeg -c:v copy → out.mp4 + vi.srt → chép lên Drive
```

Sau **mỗi** bước: ghi file `steps/<n>.done` rồi chép thay đổi lên Drive. Lần chạy sau, bước nào có file `.done` thì bỏ qua.

### 2.3 Hai chế độ đầu ra

| Chế độ | Cách làm | Cần tách nguồn? | Ghi chú |
|---|---|---|---|
| **Thuyết minh** (mặc định) | Giữ audio gốc, giảm 12–18 dB khi có giọng Việt | Không | Nhanh, không có lỗi do tách nguồn |
| Lồng tiếng | Bỏ giọng gốc, giữ nhạc nền và tiếng động | Có (GPU, ~1–3 phút) | Giống bản lồng tiếng thật. Có thể nghe thấy giọng gốc rò rỉ |

### 2.4 Cấu trúc repo

```
MinhKhoiDubDub/
├── README.md                 # nút "Open in Colab" + hướng dẫn 5 bước
├── PLAN.md
├── colab/SuperDubDub.ipynb   # notebook duy nhất, ô nào cũng ngắn: chỉ gọi vào package dub/
├── pyproject.toml            # app chính: fastapi, uvicorn, numpy, soundfile, httpx, pysubs2
├── dub/
│   ├── cli.py                # python -m dub run <video> --src ja --mode voiceover
│   ├── pipeline.py           # thứ tự bước, resume, đồng bộ lên Drive
│   ├── steps/                # mỗi bước trong §2.2 là một file
│   ├── llm.py                # Gemini (REST, structured output)
│   ├── names.py              # tiếng Trung → Hán-Việt, tiếng Nhật → Romaji
│   ├── fit.py  cues.py       # thuật toán thuần, có unit test, chạy được trên laptop
│   └── web/                  # editor: app.py + static/*.html (không cần bước build)
├── engines/{asr,sep,tts}/    # requirements + run.py (đọc JSON vào, ghi JSON/WAV ra)
├── voices/                   # giọng mẫu 6–10 giây cho viXTTS
└── tests/                    # pytest cho fit, cues, names, schema
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

`flags` chứa các cảnh báo như `"needs_review"` hoặc `"asr_low_conf"`. Editor hiển thị các segment có cờ lên đầu danh sách.

---

## 3. Lựa chọn công nghệ (tất cả miễn phí)

Các mục đánh dấu ⚖ sẽ được chốt bằng số liệu đo ở Phase 0.

### 3.1 Nhận dạng giọng nói (ASR), chạy trên GPU T4

| Ngôn ngữ | Mặc định | Dự phòng | Ghi chú |
|---|---|---|---|
| Anh | WhisperX `large-v3` fp16 | `large-v3-turbo` (nhanh hơn) | Căn chỉnh từng từ bằng wav2vec2 |
| Nhật | WhisperX `large-v3` | n/a | ⚖ Đo CER trên clip anime |
| Trung | WhisperX `large-v3` + OpenCC (đổi chữ phồn thể sang giản thể) | FunASR Paraformer-zh | ⚖ So CER giữa hai engine |

- **Chống "bịa chữ" (hallucination):** bật lọc VAD, đặt `condition_on_previous_text=False`, gắn cờ cho segment có `compression_ratio > 2.4` hoặc `avg_logprob < -1.0`.
- **Diarization:** pyannote trên GPU chạy nhanh, nhưng cần `HF_TOKEN` và phải đồng ý điều khoản model trên Hugging Face. Mặc định tắt, bật cho phim.

### 3.2 Dịch: Gemini API (free tier)

- Model Flash trong free tier, gọi bằng REST với **structured output** (JSON schema).
- **Pass 1** (một lần cho mỗi video): tóm tắt, thể loại, văn phong, nhân vật, **bảng xưng hô theo từng cặp nhân vật**, glossary. Lưu theo series để dùng lại cho các tập sau.
- **Pass 2**: mỗi chunk khoảng 40–60 segment, kèm 10 câu đã dịch trước đó làm ngữ cảnh. Kiểm tra số lượng và ID khớp 1:1. Sai thì thử lại tối đa 2 lần, sau đó gắn cờ.
- **Mỗi segment có 2 bản dịch:**
  - `sub`: tối đa 2 dòng × 42 ký tự, tốc độ đọc ≤ ~17 ký tự/giây.
  - `dub`: tối đa `syllable_budget` âm tiết, viết sẵn ở **dạng đọc** (số, ngày, đơn vị ghi bằng chữ).
- **Đếm âm tiết tiếng Việt bằng `len(text.split())`.**
  - `syllable_budget = floor(window_sec × voice_rate)`
  - `voice_rate` đo cho từng giọng ở Phase 0, thường khoảng 4.5–5.5 âm tiết/giây.
- **Tên riêng xử lý bằng code**, sau đó đưa vào glossary:
  - Tiếng Trung: tra bảng phiên âm Hán-Việt theo từng chữ (张伟 thành Trương Vĩ).
  - Tiếng Nhật: chuyển sang Romaji bằng `pykakasi`/`cutlet`.
- **Hết hạn mức free tier:** code tự chờ và thử lại theo header `retry-after`. Job dừng ở bước 6 và có thể chạy tiếp hôm sau nhờ resume. *Phương án dự phòng (chỉ làm khi thật sự cần):* chạy Ollama với `qwen2.5:7b` ngay trên GPU của Colab.
- Lưu ý: dữ liệu gửi lên Gemini free tier có thể được Google dùng để cải thiện sản phẩm.

### 3.3 Giọng đọc tiếng Việt (TTS), chạy trên GPU T4

| Ứng viên | Clone giọng | Vai trò | Rủi ro |
|---|---|---|---|
| **viXTTS** (XTTS-v2 fine-tune tiếng Việt) | Có (mẫu 6–10 giây) | **Mặc định tạm thời** | License CPML: chỉ phi thương mại (phù hợp vì dùng cá nhân). Đôi khi lặp hoặc nuốt chữ ở câu rất ngắn |
| Checkpoint F5-TTS tiếng Việt của cộng đồng | Có | ⚖ Ứng viên thay thế | CC-BY-NC (phù hợp). Chất lượng tuỳ checkpoint |
| **edge-tts** (HoaiMy, NamMinh) | Không | Xem trước nhanh, dự phòng | Endpoint không chính thức, có thể ngừng hoạt động |

⚖ **Bài nghe mù:** 20 câu × 3 engine, bạn chấm 1–5 điểm mà không biết engine nào. Code chỉ có **một hàm** `synthesize(text, voice, speed) -> wav`. Engine thứ hai chỉ được thêm vào khi bài test chứng minh cần.

- **Câu rất ngắn** (dưới 3 âm tiết): gộp với câu kề bên của cùng speaker trước khi đưa vào TTS, để tránh XTTS bị lặp hoặc nuốt chữ.
- **Kiểm tra tự động:** nếu thời lượng TTS bất thường so với số âm tiết (lớn hơn 2 lần hoặc nhỏ hơn 0.5 lần mức dự kiến) thì tạo lại câu đó tối đa 2 lần.

### 3.4 Tách nguồn, mix, mux

| Hạng mục | Công cụ | Ghi chú |
|---|---|---|
| Tách nguồn | ⚖ Demucs `htdemucs_ft` và BS-RoFormer (qua `audio-separator`), chạy GPU | Chỉ dùng cho chế độ lồng tiếng |
| Ducking | numpy: tính envelope từ timeline đã biết trước | Giảm nền 8–12 dB (lồng tiếng) hoặc 12–18 dB (thuyết minh). Fade vào 150 ms, fade ra 250 ms |
| Co giãn thời gian | Ưu tiên tham số `speed` của TTS (tạo lại câu). Dự phòng: `atempo` / `rubberband` của ffmpeg | ≤ 1.1x gần như không nghe ra |
| Chuẩn hoá âm lượng | `loudnorm` 2 lượt: `I=-16:TP=-1.5:LRA=11` | |
| Mux | `ffmpeg -c:v copy`: track tiếng Việt và track tiếng gốc + phụ đề `mov_text` | Không encode lại video |
| Burn-in phụ đề | `h264_nvenc` trên T4 (tuỳ chọn) | Nhanh vì dùng GPU encode |

---

## 4. Thuật toán (không đổi so với bản 1.0)

### 4.1 Khớp thời lượng (bước 8)

`W` là cửa sổ khả dụng (tính đến lúc câu tiếp theo bắt đầu, trừ 100 ms), `D` là thời lượng TTS đã cắt khoảng lặng ở đầu và cuối, `r = D / W`.

| r | Hành động |
|---|---|
| ≤ 1.0 | Đặt tại thời điểm bắt đầu của từ đầu tiên. Phần thừa để im lặng |
| 1.0–1.15 | Tạo lại TTS với `speed = r` |
| 1.15–1.3 | Đặt sớm hơn tối đa 200 ms, kéo dài sang gap phía sau, speed ≤ 1.2 |
| > 1.3 | Gemini rút gọn `dub` theo ngân sách âm tiết mới, rồi TTS lại. Tối đa 2 vòng |
| Vẫn không vừa | Ép tối đa 1.35x và gắn cờ `needs_review` |

Thuật toán greedy từ trái sang phải, biến `t_free` bảo đảm **không bao giờ có hai câu chồng nhau**. Viết dưới dạng hàm thuần, có unit test, **chạy được trên laptop**.

### 4.2 Luật chia cue phụ đề (bước 4)

- Mỗi cue dài 0.83–7 giây, cách nhau ≥ 2 frame, tối đa 2 dòng × 42 ký tự. Không ngắt dòng giữa một từ ghép.
- **Có 2 track phụ đề:**
  - `vi.srt`: timing theo giọng lồng tiếng, lấy từ `placed_start`.
  - `vi_orig.srt`: timing theo giọng gốc, lấy từ timestamp từng từ của ASR.

---

## 5. Quy trình phát triển

| Ở đâu | Làm gì |
|---|---|
| **Laptop** (Python 3.11 qua uv, chỉ cài thư viện nhẹ) | Viết code. Chạy `pytest` cho các phần thuần như `fit`, `cues`, `names`, schema. Không cài torch hay model nào |
| **GitHub** | Mỗi tính năng một branch, mở PR, merge vào `main` |
| **Colab** | Bạn mở notebook, đặt `BRANCH = "tên-branch"` ở ô 1 để thử code mới, bấm chạy, rồi gửi lỗi (nếu có) cho Claude |

Claude không chạy được Colab. Phần chạy với GPU luôn cần bạn bấm chạy và báo lại kết quả.

---

## 6. Lộ trình triển khai

### Phase 0: Khung notebook và benchmark (1 tuần)

**Việc cần làm:**
1. README có nút "Open in Colab". Notebook ô 1 gồm: mount Drive, clone repo theo `BRANCH`, kiểm tra GPU, cài đặt bằng uv.
2. Đo thời gian cài đặt mỗi phiên. **Mục tiêu dưới 7 phút**, nếu vượt thì tìm cách cache.
3. Benchmark trên 5 video SQL có sẵn và các clip tiếng Nhật, tiếng Trung, phim tiếng Anh mà bạn thêm vào `Drive/SuperDubDub/input`:
   - ASR: WER (tiếng Anh) và CER (tiếng Nhật, tiếng Trung), RTF, VRAM đỉnh.
   - Gemini: chất lượng dịch trên 50 câu, số request đã dùng so với hạn mức.
   - TTS: bài nghe mù giữa viXTTS, F5-VN và edge-tts. Đo `voice_rate`.
   - Tách nguồn: Demucs so với RoFormer.
4. Ghi kết quả vào `BENCHMARK.md`.

**Tiêu chí hoàn thành:**
- Notebook chạy được từ máy trắng.
- Đã chốt các mục ⚖.
- Biết chính xác thời gian cài đặt và tốc độ của từng bước.

### Phase 1: Pipeline chạy hết trên Colab (2 tuần)

**Việc cần làm:**
- Các bước 1, 3, 4, 5, 6, 7, 9, 10 ở chế độ thuyết minh.
- Đồng bộ lên Drive và resume.
- Unit test chạy trên laptop.

**Tiêu chí hoàn thành:**
- Video SQL 20 phút ("CREATE Table") chạy hết mà không cần can thiệp.
- Ngắt runtime giữa chừng, chạy lại thì tiếp tục đúng bước.
- `out.mp4` trên Drive phát được và chuyển được giữa 2 track audio.

### Phase 2: Đồng bộ và chất lượng âm thanh (1–2 tuần)

**Việc cần làm:**
- Fit solver và vòng rút gọn câu bằng Gemini.
- Luật chia cue, ducking, loudnorm.
- Bước 2 (tách nguồn) và chế độ lồng tiếng.

**Tiêu chí hoàn thành:**
- Dưới 5% segment có `rate > 1.15`.
- Không có câu nào chồng nhau.
- Loudness đạt −16 ± 1 LUFS.

### Phase 3: Editor trong notebook (2 tuần)

**Việc cần làm:**
- FastAPI chạy nền, hiển thị bằng iframe trong notebook.
- Giao diện: video player cạnh bảng segment, bấm dòng nào thì video tua đến đó, sửa được `sub`, `dub`, speaker và giọng.
- Nút "Nghe thử câu này", bộ lọc segment có cờ, sửa glossary và bảng xưng hô.
- Nút **Tạo lại** chỉ chạy lại các segment đã sửa.

**Tiêu chí hoàn thành:** sửa 1 câu và xuất lại video trong dưới 1 phút.

### Phase 4: Nâng cao (tuỳ chọn)

- Diarization và clone giọng từng nhân vật từ chính video.
- Bộ nhớ theo series.
- Burn-in phụ đề bằng NVENC.
- Chạy nhiều video liên tiếp trong một phiên.
- Dự phòng bằng Kaggle khi Colab không cấp GPU.

**Tổng thời gian:** khoảng 6–8 tuần bán thời gian.

---

## 7. Ước lượng hiệu năng trên T4 (sẽ thay bằng số đo thật)

Video 30 phút:

| Bước | Thời gian |
|---|---|
| Cài đặt môi trường và tải model (mỗi phiên) | 5–7 phút |
| Ingest | < 1 phút |
| ASR large-v3 + căn chỉnh từng từ | 3–6 phút |
| Dịch bằng Gemini (2 pass, có thể phải chờ hạn mức) | 3–8 phút |
| TTS viXTTS (~20 phút giọng nói) | 6–12 phút |
| Fit, mix, mux, chép lên Drive | 3–5 phút |
| **Tổng** | **~20–35 phút** (cộng 1–3 phút nếu tách nguồn) |

---

## 8. Rủi ro và cách giảm thiểu

| Rủi ro | Khả năng | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| Colab không cấp GPU, hoặc hết hạn mức dùng | Trung bình | Cao | Resume, đổi giờ chạy. Phase 4 có dự phòng Kaggle |
| Vi phạm điều khoản free tier của Colab | Thấp nếu làm đúng | Cao (bị khoá tài khoản) | Không tunnel, không link public, editor chỉ hiện trong notebook |
| Hết hạn mức Gemini free tier | Trung bình | Trung bình | Chunk lớn để giảm số request, chờ theo `retry-after`, resume hôm sau. Dự phòng Ollama trên Colab |
| Ảnh môi trường Colab cập nhật làm hỏng cài đặt | Trung bình | Trung bình | Venv riêng, khoá phiên bản. Ô cài đặt kiểm tra phiên bản trước khi chạy |
| Ngắt phiên giữa chừng | Cao | Thấp | Lưu lên Drive sau mỗi bước, resume |
| Drive đầy (15 GB) | Trung bình | Trung bình | Tự dọn file trung gian, cảnh báo khi Drive còn dưới 2 GB |
| viXTTS lặp hoặc nuốt chữ | Trung bình | Thấp | Gộp câu ngắn, kiểm tra thời lượng bất thường rồi tạo lại |
| Dịch sai xưng hô | Trung bình | Cao | Bảng xưng hô ở pass 1, editor sửa tay, bộ nhớ theo series |
| Bản quyền video nguồn | n/a | Pháp lý | Chỉ dùng cá nhân. Drive để chế độ riêng tư. Không đăng lại |

---

## 9. Chỉ số chất lượng

- **ASR:** WER/CER, % segment bị gắn cờ.
- **Dịch:** **tỷ lệ câu phải sửa tay** trong editor (chỉ số quan trọng nhất).
- **Đồng bộ:** phân bố `rate`, % `needs_review`, số câu chồng nhau (phải bằng 0).
- **Âm thanh:** LUFS, true peak.
- **Vận hành:** thời gian cài đặt, tổng thời gian xử lý mỗi phút video, số lần chạm hạn mức Gemini.

---

## 10. Việc bạn cần chuẩn bị trước Phase 0

1. Lưu Gemini API key vào **Colab Secrets** với tên `GEMINI_API_KEY`, và bật quyền truy cập cho notebook.
2. Tạo thư mục `MyDrive/SuperDubDub/input` rồi upload video test vào đó. Ngoài 5 video SQL đã có, cần thêm clip tiếng Nhật, tiếng Trung và phim tiếng Anh.
3. *(Tuỳ chọn, cho diarization)* Tạo token Hugging Face (loại read), lưu vào Secrets với tên `HF_TOKEN`, và bấm đồng ý điều khoản của model `pyannote/speaker-diarization-3.1`.
