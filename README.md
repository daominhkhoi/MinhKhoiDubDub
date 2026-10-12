# MinhKhoiDubDub

Lồng tiếng và phụ đề video nước ngoài (Anh, Nhật, Trung) sang tiếng Việt. Miễn phí 100%, chạy trên Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daominhkhoi/MinhKhoiDubDub/blob/main/colab/SuperDubDub.ipynb)

## Dùng thế nào

1. Upload video vào `Google Drive/SuperDubDub/input/`.
2. Bấm nút **Open in Colab** ở trên.
3. `Runtime → Change runtime type → T4 GPU`.
4. Biểu tượng 🔑 bên trái → thêm secret `GEMINI_API_KEY` (lấy miễn phí tại aistudio.google.com), bật **Notebook access**.
5. Chạy lần lượt các ô.

Hiện đang ở **Phase 0 (benchmark)**. Kế hoạch đầy đủ ở [PLAN.md](PLAN.md).

## Phát triển

Code AI chạy trên Colab. Trên máy local chỉ chạy unit test cho phần thuật toán thuần (không cần GPU hay model):

```bash
uv run --python 3.11 pytest
```

| Thư mục | Nội dung |
|---|---|
| `colab/` | Notebook và script cài đặt trên Colab |
| `dub/` | Package chính: dịch (Gemini), benchmark, sau này là pipeline |
| `engines/<tên>/` | Mỗi engine AI một venv riêng (`run.py` + `requirements.txt`), gọi qua `engines/run.sh` |
| `tests/` | Unit test chạy được trên laptop |
