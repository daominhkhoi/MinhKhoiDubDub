"""Gọi Gemini API (free tier) và nhận JSON theo schema."""
import json
import os
import time

import requests

API = "https://generativelanguage.googleapis.com/v1beta"
calls = 0  # số request đã gửi, để đối chiếu với hạn mức free tier


def _retry_delay(resp, default):
    """Số giây cần chờ theo RetryInfo của Google, nếu có."""
    try:
        for d in resp.json()["error"]["details"]:
            if d.get("@type", "").endswith("RetryInfo"):
                return float(d["retryDelay"].rstrip("s")) + 1
    except (ValueError, KeyError, TypeError):
        pass
    return default


def flash_models(key):
    r = requests.get(f"{API}/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    r.raise_for_status()
    return [m["name"].removeprefix("models/") for m in r.json().get("models", []) if "flash" in m["name"]]


def generate_json(prompt, schema, *, system=None, model=None, tries=5):
    global calls
    key = os.environ["GEMINI_API_KEY"]
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0.3},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    for i in range(tries):
        calls += 1
        r = requests.post(f"{API}/models/{model}:generateContent", headers={"x-goog-api-key": key}, json=body, timeout=300)
        if r.status_code in (429, 500, 503) and i < tries - 1:
            time.sleep(_retry_delay(r, 5 * 2**i))
            continue
        if r.status_code == 404:
            raise RuntimeError(f"Không có model '{model}'. Các model Flash khả dụng: {flash_models(key)}")
        r.raise_for_status()
        cand = r.json()["candidates"][0]
        parts = cand.get("content", {}).get("parts")
        if not parts:
            raise RuntimeError(f"Gemini không trả nội dung (finishReason={cand.get('finishReason')})")
        return json.loads("".join(p.get("text", "") for p in parts))
