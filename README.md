# MinhKhoiDubDub

Lồng tiếng và phụ đề video nước ngoài (Anh, Nhật, Trung) sang tiếng Việt. Miễn phí 100%, chạy trên GPU miễn phí của Kaggle.

Hiện đang ở **Phase 0 (benchmark)**. Kế hoạch đầy đủ ở [PLAN.md](PLAN.md).

## Chạy trên Kaggle (từ laptop)

Cần: tài khoản Kaggle đã xác minh số điện thoại, Kaggle API token, secret `GEMINI_API_KEY` gắn vào notebook `superdubdub-bench` (Add-ons → Secrets).

```bash
kaggle kernels push -p kaggle
```

```bash
kaggle kernels status khiominhk19fptuhcm/superdubdub-bench
```

```bash
kaggle kernels output khiominhk19fptuhcm/superdubdub-bench -p out
```

Kết quả trong `out/`: `report.json` (số liệu), `logs/` (log từng engine), `listen/` (file nghe mù X/Y/Z).

## Phát triển

Trên laptop chỉ chạy unit test cho phần thuật toán thuần (không cần GPU hay model):

```bash
uv run --python 3.11 pytest
```

| Thư mục | Nội dung |
|---|---|
| `kaggle/` | Kernel Kaggle: clone repo, cài engine, chạy benchmark |
| `dub/` | Package chính: dịch (Gemini), benchmark, sau này là pipeline |
| `engines/<tên>/` | Mỗi engine AI một venv riêng (`run.py` + `requirements.txt`), cài bằng `engines/setup.sh`, gọi qua `engines/run.sh` |
| `tests/` | Unit test chạy được trên laptop |
