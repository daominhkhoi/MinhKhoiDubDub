import pytest

from dub import llm
from dub.translate import budgets, check, syllables


def test_syllables_counts_vietnamese_words():
    assert syllables("Tớ tuyệt đối  không bỏ cuộc!") == 6


def test_budgets_use_gap_to_next_segment():
    segs = [{"start": 0.0, "end": 1.5}, {"start": 2.1, "end": 3.0}]
    # câu 1: (2.1 - 0.1 - 0) * 5 = 10; câu cuối: (3.0 + 1.0 - 2.1) * 5 = 9.5 -> 9
    assert budgets(segs, 5.0) == [10, 9]


def test_budgets_never_below_two():
    segs = [{"start": 0.0, "end": 0.1}, {"start": 0.15, "end": 0.2}]
    assert budgets(segs, 5.0)[0] == 2


def test_check_rejects_merged_or_reordered_ids():
    check([3, 4], [{"id": 3}, {"id": 4}])
    with pytest.raises(ValueError):
        check([3, 4], [{"id": 3}])
    with pytest.raises(ValueError):
        check([3, 4], [{"id": 4}, {"id": 3}])


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_generate_json_retries_after_rate_limit(monkeypatch):
    ok = {"candidates": [{"content": {"parts": [{"text": '{"a": 1}'}]}}]}
    limited = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}]}}
    replies = [FakeResp(429, limited), FakeResp(200, ok)]
    slept = []
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    assert llm.generate_json("p", {}) == {"a": 1}
    assert slept == [3.0]


def test_generate_json_explains_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: FakeResp(200, {"candidates": [{"finishReason": "SAFETY"}]}))
    with pytest.raises(RuntimeError, match="SAFETY"):
        llm.generate_json("p", {})
