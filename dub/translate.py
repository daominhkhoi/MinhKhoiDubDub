"""Dịch 2 pass: pass 1 lấy ngữ cảnh cả video, pass 2 dịch từng chunk ra {sub, dub}."""
import json
import math

from dub.llm import generate_json

LANGS = {"en": "tiếng Anh", "ja": "tiếng Nhật", "zh": "tiếng Trung"}

SYSTEM = """Bạn là biên dịch viên phụ đề và lồng tiếng chuyên nghiệp, dịch sang tiếng Việt tự nhiên như người Việt nói.
Quy tắc:
- Xưng hô theo đúng quan hệ nhân vật trong bảng xưng hô; giữ nhất quán suốt video.
- Tên riêng và thuật ngữ theo glossary. Thuật ngữ kỹ thuật phổ biến (SQL, database, query...) có thể giữ nguyên tiếng Anh nếu người Việt thường nói vậy.
- "sub": phụ đề để đọc, tối đa 2 dòng x 42 ký tự, súc tích.
- "dub": câu để lồng tiếng, KHÔNG vượt quá max_syll âm tiết (đếm theo khoảng trắng), viết ở dạng đọc:
  số, ngày, đơn vị, ký hiệu viết bằng chữ (ví dụ "2022" -> "hai nghìn không trăm hai mươi hai").
- Không gộp, không tách, không bỏ câu: mỗi id đầu vào có đúng một kết quả."""

CONTEXT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "genre": {"type": "STRING"},
        "register": {"type": "STRING"},
        "characters": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "name": {"type": "STRING"}, "gender": {"type": "STRING"}, "note": {"type": "STRING"}}}},
        "pronouns": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "speaker": {"type": "STRING"}, "listener": {"type": "STRING"},
            "self": {"type": "STRING"}, "other": {"type": "STRING"}}}},
        "glossary": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "src": {"type": "STRING"}, "vi": {"type": "STRING"}}}},
    },
    "required": ["summary", "genre", "register", "characters", "pronouns", "glossary"],
}

CHUNK_SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "id": {"type": "INTEGER"}, "sub": {"type": "STRING"}, "dub": {"type": "STRING"}},
        "required": ["id", "sub", "dub"]},
}


def syllables(text):
    """Tiếng Việt đơn âm tiết: số âm tiết = số cụm tách theo khoảng trắng."""
    return len(text.split())


def budgets(segments, voice_rate, gap=0.1, tail=1.0):
    """Số âm tiết tối đa cho mỗi câu: được nói đến khi câu sau bắt đầu (trừ gap)."""
    out = []
    for i, s in enumerate(segments):
        end = segments[i + 1]["start"] - gap if i + 1 < len(segments) else s["end"] + tail
        out.append(max(2, math.floor((end - s["start"]) * voice_rate)))
    return out


def check(chunk_ids, result):
    """Kết quả phải khớp 1:1 với id đầu vào, đúng thứ tự."""
    got = [r["id"] for r in result]
    if got != chunk_ids:
        raise ValueError(f"id không khớp: gửi {chunk_ids}, nhận {got}")


def context_pass(segments, src_lang):
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(segments))
    prompt = (f"Transcript {LANGS.get(src_lang, src_lang)} của một video. Phân tích để chuẩn bị dịch sang tiếng Việt: "
              f"tóm tắt, thể loại, văn phong, nhân vật, bảng xưng hô từng cặp, glossary tên riêng/thuật ngữ.\n\n{lines}")
    return generate_json(prompt, CONTEXT_SCHEMA, system=SYSTEM)


def translate(segments, src_lang, context, voice_rate=5.0, chunk=50, prev_n=10, tries=3):
    """Trả về list {id, sub, dub, max_syll} theo thứ tự segments."""
    limit = budgets(segments, voice_rate)
    done = []
    for start in range(0, len(segments), chunk):
        ids = list(range(start, min(start + chunk, len(segments))))
        rows = [{"id": i, "text": segments[i]["text"], "max_syll": limit[i]} for i in ids]
        prompt = (f"Ngữ cảnh video:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                  f"Các câu đã dịch ngay trước (để giữ mạch):\n{json.dumps(done[-prev_n:], ensure_ascii=False)}\n\n"
                  f"Dịch các câu {LANGS.get(src_lang, src_lang)} sau:\n{json.dumps(rows, ensure_ascii=False)}")
        for attempt in range(tries):
            result = generate_json(prompt, CHUNK_SCHEMA, system=SYSTEM)
            try:
                check(ids, result)
                break
            except ValueError:
                if attempt == tries - 1:
                    raise
        done += [{**r, "max_syll": limit[r["id"]]} for r in result]
    return done
