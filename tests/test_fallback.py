"""The provider chain: Gemini first, then Groq, then a runner-local Ollama.

Ported from sfevents, where Gemini 503'd every batch of a scheduled run.
"""
import io
import json
import re
import urllib.error

import pytest

from technews import rank


def _row(i):
    return {"id": i, "title": f"Item{i}", "source": "hackernews"}


def _scores_all(prompt, model, key, timeout):
    batch = re.findall(r"^(\d+)\.\s", prompt, re.MULTILINE)
    return json.dumps([{"i": int(i), "score": 70, "reason": model} for i in batch])


def _unavailable(prompt, model, key, timeout):
    raise rank.ProviderError("HTTP 503: high demand", status=503)


def _two_providers(monkeypatch, primary, fallback, fallback_key="y"):
    monkeypatch.setitem(rank.PROVIDERS, "first", ("FIRST_KEY", "first-m", primary))
    monkeypatch.setitem(rank.PROVIDERS, "backup", ("BACKUP_KEY", "backup-m", fallback))
    monkeypatch.setitem(rank.FALLBACK_PACING, "backup", (2, 30.0))
    monkeypatch.setenv("FIRST_KEY", "x")
    if fallback_key:
        monkeypatch.setenv("BACKUP_KEY", fallback_key)
    else:
        monkeypatch.delenv("BACKUP_KEY", raising=False)
    monkeypatch.setattr(rank, "RETRY_WAITS", (0, 0, 0))


def test_what_the_primary_misses_goes_to_the_fallback(monkeypatch):
    _two_providers(monkeypatch, _unavailable, _scores_all)
    out = rank.llm_scores_chain([_row(i) for i in range(1, 5)], "p", ["first", "backup"],
                                sleep=lambda s: None)
    assert set(out) == {"backup:backup-m"}
    assert set(out["backup:backup-m"]) == {1, 2, 3, 4}


def test_the_fallback_only_sees_items_the_primary_missed(monkeypatch):
    calls = {"n": 0}

    def first_batch_only(prompt, model, key, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps([{"i": 1, "score": 10, "reason": "ok"},
                               {"i": 2, "score": 20, "reason": "ok"}])
        raise rank.ProviderError("HTTP 400: bad", status=400)

    sent = []

    def backup(prompt, model, key, timeout):
        sent.append(prompt)
        return _scores_all(prompt, model, key, timeout)

    _two_providers(monkeypatch, first_batch_only, backup)
    out = rank.llm_scores_chain([_row(i) for i in range(1, 5)], "p", ["first", "backup"],
                                batch_size=2, sleep=lambda s: None)
    assert set(out["first:first-m"]) == {1, 2}
    assert set(out["backup:backup-m"]) == {3, 4}
    assert all("Item1 " not in p and "Item2 " not in p for p in sent)


def test_the_fallback_is_paced_by_its_own_limits(monkeypatch):
    _two_providers(monkeypatch, _unavailable, _scores_all)
    waits = []
    clock = iter(range(0, 1000))
    out = rank.llm_scores_chain([_row(i) for i in range(1, 7)], "p", ["first", "backup"],
                                batch_size=50, sleep=waits.append, clock=lambda: next(clock))
    assert len(out["backup:backup-m"]) == 6
    assert [w for w in waits if w > 0] == [29, 29]


def test_a_fallback_without_a_key_is_skipped_not_fatal(monkeypatch):
    _two_providers(monkeypatch, _unavailable, _scores_all, fallback_key=None)
    notes = []
    out = rank.llm_scores_chain([_row(1)], "p", ["first", "backup"], sleep=lambda s: None,
                                on_provider=lambda n, m, k, note: notes.append(note))
    assert out == {}
    assert "BACKUP_KEY not set" in notes[-1]


def test_nothing_goes_to_the_fallback_when_the_primary_succeeds(monkeypatch):
    _two_providers(monkeypatch, _scores_all,
                   lambda *a: pytest.fail("fallback should not be called"))
    out = rank.llm_scores_chain([_row(1)], "p", ["first", "backup"])
    assert set(out) == {"first:first-m"}


def test_a_dropped_connection_is_retried_like_a_503(monkeypatch):
    calls = {"n": 0}

    def reset_once(prompt, model, key, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError(ConnectionResetError(104, "reset by peer"))
        return _scores_all(prompt, model, key, timeout)

    _two_providers(monkeypatch, reset_once, _unavailable)
    out = rank.llm_scores_chain([_row(1)], "p", ["first"], sleep=lambda s: None)
    assert set(out["first:first-m"]) == {1}


def test_ollama_is_the_last_resort_and_gets_a_long_timeout(monkeypatch):
    timeouts = []

    def local(prompt, model, host, timeout):
        timeouts.append(timeout)
        return _scores_all(prompt, model, host, timeout)

    _two_providers(monkeypatch, _unavailable, _unavailable)
    monkeypatch.setitem(rank.PROVIDERS, "ollama", ("OLLAMA_HOST", "tiny", local))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    out = rank.llm_scores_chain([_row(i) for i in range(1, 13)], "p",
                                ["first", "backup", "ollama"], sleep=lambda s: None)
    assert set(out) == {"ollama:tiny"}
    assert len(out["ollama:tiny"]) == 12
    assert timeouts == [600, 600]


def test_groq_sends_an_openai_style_request(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured.update(url=req.full_url, auth=req.headers.get("Authorization"),
                        body=json.loads(req.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "[]"}}]}).encode())

    monkeypatch.setattr(rank.urllib.request, "urlopen", fake_urlopen)
    assert rank._call_groq("hello", rank.GROQ_MODEL, "gsk_test", 30) == "[]"
    assert captured["url"] == rank.GROQ_URL
    assert captured["auth"] == "Bearer gsk_test"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_a_groq_daily_limit_is_recognised(monkeypatch):
    body = (b'{"error":{"message":"Rate limit reached for model openai/gpt-oss-120b '
            b'on tokens per day (TPD): Limit 200000, Used 199000"}}')

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(body))

    monkeypatch.setattr(rank.urllib.request, "urlopen", boom)
    with pytest.raises(rank.ProviderError) as info:
        rank._post_json(rank.GROQ_URL, {}, {}, 30)
    assert info.value.daily


def test_ollama_asks_for_a_big_enough_context_and_no_thinking(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured.update(url=req.full_url, body=json.loads(req.data))
        return io.BytesIO(json.dumps({"message": {"content": "[]"}}).encode())

    monkeypatch.setattr(rank.urllib.request, "urlopen", fake_urlopen)
    rank._call_ollama("hello", rank.OLLAMA_MODEL, "127.0.0.1:11434", 600)
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    body = captured["body"]
    assert body["stream"] is False and body["think"] is False
    assert body["options"]["num_ctx"] >= 8192


def test_requests_do_not_use_urllibs_default_user_agent(monkeypatch):
    """Groq's Cloudflare front door 403s "Python-urllib/3.x" (error 1010)."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        return io.BytesIO(b"{}")

    monkeypatch.setattr(rank.urllib.request, "urlopen", fake_urlopen)
    rank._post_json(rank.GROQ_URL, {"Authorization": "Bearer x"}, {}, 30)
    assert captured["ua"] and "Python-urllib" not in captured["ua"]
