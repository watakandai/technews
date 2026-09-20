import json
from datetime import datetime, timedelta, timezone

import pytest

from technews import rank as ranking
from technews.rank import (
    ProviderError, cluster_sources, freshness, heuristic_scores, hours_old,
    llm_scores, parse_results, profile_hash, strip_comments,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def row(item_id=1, source="hackernews", popularity=None, title="A story",
        hours=1.0, cluster="u:x", summary="s"):
    return {
        "id": item_id, "source": source, "popularity": popularity, "title": title,
        "cluster_key": cluster, "summary": summary,
        "published_ts": (NOW - timedelta(hours=hours)).isoformat(),
    }


# -- heuristic --------------------------------------------------------------

def test_freshness_halves_every_thirty_six_hours():
    assert freshness(0) == 1.0
    assert freshness(36) == pytest.approx(0.5)
    assert freshness(72) == pytest.approx(0.25)


def test_hours_old_falls_back_when_there_is_no_usable_date():
    assert hours_old({"published_ts": None, "first_seen": None}) == 48.0
    assert hours_old({"published_ts": "not a date", "first_seen": None}) == 48.0


def test_a_fresh_popular_story_outranks_an_old_one():
    fresh = heuristic_scores([row(1, popularity=90, hours=2)], now=NOW)[1][0]
    stale = heuristic_scores([row(1, popularity=90, hours=200)], now=NOW)[1][0]
    assert fresh > stale


def test_items_without_popularity_start_from_a_neutral_prior_not_zero():
    # "No source counted votes" is not evidence of being uninteresting.
    score = heuristic_scores([row(1, source="arxiv", popularity=None)], now=NOW)[1][0]
    assert 20 < score < 60


def test_cross_source_agreement_raises_the_score():
    alone = heuristic_scores([row(1, cluster="u:a")], now=NOW)[1][0]
    both = heuristic_scores(
        [row(1, cluster="u:a"), row(2, source="reddit", cluster="u:a")], now=NOW)[1][0]
    assert both > alone
    assert "2 sources" in heuristic_scores(
        [row(1, cluster="u:a"), row(2, source="reddit", cluster="u:a")], now=NOW)[1][1]


def test_recurring_threads_are_pushed_down():
    normal = heuristic_scores([row(1, title="A real story")], now=NOW)[1][0]
    filler = heuristic_scores([row(1, title="Ask HN: Who is hiring?")], now=NOW)[1][0]
    assert filler < normal


def test_cluster_sources_groups_by_key_and_leaves_singletons_alone():
    rows = [row(1, cluster="u:a"), row(2, source="reddit", cluster="u:a"),
            row(3, source="lobsters", cluster="")]
    grouped = cluster_sources(rows)
    assert grouped[1] == {"hackernews", "reddit"}
    assert grouped[3] == {"lobsters"}


def test_scores_stay_inside_the_scale():
    rows = [row(1, popularity=100, hours=0, source="hackernews"),
            row(2, popularity=0, hours=999, source="bluesky", title="sponsored deals roundup")]
    for score, _ in heuristic_scores(rows, now=NOW).values():
        assert 0.0 <= score <= 100.0


# -- profile ----------------------------------------------------------------

def test_html_comments_never_reach_the_model():
    cleaned = strip_comments("Real text\n<!-- edit this file, it is a template -->\nMore")
    assert "template" not in cleaned and "Real text" in cleaned


def test_profile_hash_changes_with_the_profile_or_the_model():
    assert profile_hash("a", "m") != profile_hash("b", "m")
    assert profile_hash("a", "m") != profile_hash("a", "other")


def test_profile_hash_changes_with_the_taxonomy(monkeypatch):
    # A cached row holds a category as well as a score, so splitting a
    # category has to invalidate the cache the same way editing the profile
    # does - otherwise old items stay filed under a bucket that moved.
    before = profile_hash("a", "m")
    monkeypatch.setattr(ranking, "CATEGORY_LIST", "robotics (Robots), other (Rest)")
    assert profile_hash("a", "m") != before


# -- reply parsing ----------------------------------------------------------

def test_parse_results_tolerates_fences_and_prose():
    reply = 'Sure!\n```json\n[{"i":1,"score":88,"category":"robotics","reason":"fits"}]\n```'
    assert parse_results(reply, 1) == {
        1: {"score": 88.0, "category": "robotics", "reason": "fits"}}


def test_parse_results_drops_out_of_range_indexes_and_invented_categories():
    reply = json.dumps([
        {"i": 1, "score": 50, "category": "not_a_real_category", "reason": "x"},
        {"i": 9, "score": 50, "category": "robotics", "reason": "out of range"},
        {"i": 2, "score": 500, "category": "ai_ml", "reason": "clamped"},
    ])
    out = parse_results(reply, 2)
    assert set(out) == {1, 2}
    assert out[1]["category"] == "", "an off-list category must not reach the filter UI"
    assert out[2]["score"] == 100.0


def test_parse_results_raises_when_there_is_no_array_at_all():
    with pytest.raises(ValueError):
        parse_results("I can't help with that.", 3)


# -- provider loop ----------------------------------------------------------

@pytest.fixture()
def fake_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def install(responder):
        def call(prompt, model, key, timeout):
            calls.append(prompt)
            return responder(len(calls), prompt)
        monkeypatch.setitem(ranking.PROVIDERS, "gemini", ("GEMINI_API_KEY", "m", call))
        return calls
    return install


def scored_reply(n_items):
    return json.dumps([{"i": i, "score": 70, "category": "ai_ml", "reason": "ok"}
                       for i in range(1, n_items + 1)])


def test_llm_scores_batches_and_maps_results_back_to_row_ids(fake_provider):
    fake_provider(lambda n, prompt: scored_reply(2))
    rows = [row(10), row(11), row(12), row(13)]
    out = llm_scores(rows, "profile", provider="gemini", batch_size=2)
    assert set(out) == {10, 11, 12, 13}
    assert out[10]["score"] == 70.0


def test_one_failing_batch_does_not_lose_the_others(fake_provider):
    fake_provider(lambda n, prompt: "garbage" if n == 1 else scored_reply(2))
    rows = [row(i) for i in range(1, 5)]
    notes = []
    out = llm_scores(rows, "p", provider="gemini", batch_size=2,
                     on_progress=lambda o, s, note: notes.append(note))
    assert set(out) == {3, 4}
    assert "FAILED" in notes[0] and "2 scored" in notes[1]


def test_a_daily_quota_stops_the_run_instead_of_burning_the_rest(fake_provider):
    def responder(n, prompt):
        raise ProviderError("quota exhausted", status=429, daily=True)
    fake_provider(responder)
    notes = []
    out = llm_scores([row(i) for i in range(1, 7)], "p", provider="gemini", batch_size=2,
                     on_progress=lambda o, s, note: notes.append(note))
    assert out == {}
    assert notes.count("skipped (rate limited)") == 2, "later batches skipped, not retried"


def test_a_per_minute_limit_is_waited_out_and_retried(fake_provider):
    state = {"n": 0}

    def responder(n, prompt):
        state["n"] += 1
        if state["n"] == 1:
            raise ProviderError("slow down", status=429, retry_after=7)
        return scored_reply(1)
    fake_provider(responder)
    waits = []
    out = llm_scores([row(1)], "p", provider="gemini", batch_size=1, sleep=waits.append)
    assert out[1]["score"] == 70.0
    assert waits == [7], "the provider's own retry-after is respected"


def test_min_interval_spaces_requests_out(fake_provider):
    fake_provider(lambda n, prompt: scored_reply(1))
    waits, clock = [], iter([0, 0, 1, 1, 2, 2, 3, 3])
    llm_scores([row(1), row(2)], "p", provider="gemini", batch_size=1,
               min_interval=10, sleep=waits.append, clock=lambda: next(clock))
    assert waits and waits[0] > 0


def test_a_missing_key_is_a_clear_error_not_a_silent_empty_run(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm_scores([row(1)], "p", provider="gemini")


def test_an_unknown_provider_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown provider"):
        llm_scores([row(1)], "p", provider="nope")


def test_the_prompt_carries_the_profile_the_categories_and_the_items(fake_provider):
    calls = fake_provider(lambda n, prompt: scored_reply(1))
    llm_scores([row(1, title="Quadruped control")], "I LIKE ROBOTS", provider="gemini")
    prompt = calls[0]
    assert "I LIKE ROBOTS" in prompt
    assert "robotics (Robotics & Autonomy)" in prompt
    assert "1. Quadruped control" in prompt


def test_api_keys_are_redacted_from_error_text():
    assert "secret" not in ranking._redact('error at ...?key=secret123&x=1')
