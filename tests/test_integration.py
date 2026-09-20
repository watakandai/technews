"""The whole pipeline, offline: fetch -> rank -> export, with fake sources."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from tech_news_curator import rank as ranking
from tech_news_curator.categorize import categorize
from tech_news_curator.db import (
    init_db, query_items, row_to_dict, set_categories, set_heuristic_scores,
    set_llm_results, set_popularity, unscored_items, upsert_items,
)
from tech_news_curator.models import Item
from tech_news_curator.popularity import popularity_scores, propagate
from tech_news_curator.cli import collapse


@pytest.fixture()
def seeded(tmp_path):
    db = tmp_path / "news.db"
    init_db(db)
    now = datetime.now(timezone.utc)
    upsert_items(db, [
        # The same article, from three places, spelled three ways.
        Item(source="hackernews", source_id="h1", title="New quadruped controller",
             url="https://ex.com/robot?utm_source=hn", discussion_url="https://news.yc/1",
             points=900, comments=300, metric="points", rank=1, published=now),
        Item(source="reddit", source_id="r1", title="New quadruped controller released",
             url="https://www.ex.com/robot/", discussion_url="https://reddit.com/r/robotics/1",
             metric="rank", rank=2, published=now),
        Item(source="ieee_spectrum", source_id="i1", title="New quadruped controller",
             url="https://ex.com/robot", summary="A long writeup of the controller.",
             published=now),
        # Unrelated, unpopular, and exactly this reader's thing.
        Item(source="arxiv", source_id="a1", title="Lyapunov certificates via SMT",
             url="http://arxiv.org/abs/2609.1", published=now - timedelta(hours=5)),
        Item(source="hackernews", source_id="h2", title="Ask HN: Who is hiring?",
             url="", discussion_url="https://news.yc/2", points=400, metric="points",
             rank=2, published=now),
    ])
    return db


def run_ranking(db):
    rows = [row_to_dict(r) for r in query_items(db)]
    scores = propagate(rows, popularity_scores(rows))
    set_popularity(db, scores)
    set_categories(db, {r["id"]: categorize(r, {"ieee_spectrum": "robotics"}) for r in rows})
    rows = [row_to_dict(r) for r in query_items(db)]
    set_heuristic_scores(db, ranking.heuristic_scores(rows))
    return [row_to_dict(r) for r in query_items(db)]


def test_the_three_copies_become_one_row_carrying_all_the_links(seeded):
    rows = run_ranking(seeded)
    merged = collapse(rows)
    quadruped = [m for m in merged if "quadruped" in m["title"]]
    assert len(quadruped) == 1, "one story, however it was spelled or tracked"
    best = quadruped[0]
    # Whichever copy represents it, all three sources are reachable from the
    # single row - none is dropped on the floor by the merge.
    assert {best["source"]} | {a["source"] for a in best["also"]} == \
        {"hackernews", "reddit", "ieee_spectrum"}


def test_the_publishers_copy_inherits_the_aggregators_attention(seeded):
    # Keyed by source_id, not source: two of these rows are both Hacker News.
    rows = {r["source_id"]: r for r in run_ranking(seeded)}
    assert rows["i1"]["popularity"] == rows["h1"]["popularity"]
    assert "via hackernews" in rows["i1"]["popularity_reason"]
    assert rows["a1"]["popularity"] is None, "a lone paper inherits nothing"


def test_a_recurring_thread_loses_to_a_real_story_despite_its_score(seeded):
    rows = {r["source_id"]: r for r in run_ranking(seeded)}
    assert rows["h2"]["score"] < rows["h1"]["score"]


def test_the_heuristic_alone_cannot_rank_a_quiet_paper_up_but_the_llm_can(seeded, monkeypatch):
    rows = {r["source_id"]: r for r in run_ranking(seeded)}
    assert rows["a1"]["score"] < rows["h1"]["score"], \
        "with no crowd signal the heuristic has nothing to go on - that is why the LLM exists"

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setitem(ranking.PROVIDERS, "gemini", ("GEMINI_API_KEY", "m", _paper_lover))
    todo = [row_to_dict(r) for r in unscored_items(seeded, "hash1")]
    results = ranking.llm_scores(todo, "I like formal methods", provider="gemini")
    set_llm_results(seeded, results, "gemini:m", "hash1")

    ranked = sorted((row_to_dict(r) for r in query_items(seeded)),
                    key=lambda r: -(r["score"] or 0))
    assert ranked[0]["source_id"] == "a1", "the profile, not the crowd, decides the top of the page"
    assert ranked[0]["category"] == "research"


def _paper_lover(prompt, model, key, timeout):
    """A stand-in model that only likes papers, so the effect is unambiguous."""
    lines = prompt.split("ITEMS\n")[1].strip().splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        paper = "Lyapunov" in line
        out.append({"i": i, "score": 98 if paper else 20,
                    "category": "research" if paper else "other",
                    "reason": "formal methods" if paper else "not their thing"})
    return json.dumps(out)


def test_a_second_run_pays_for_nothing_it_has_already_scored(seeded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    calls = []

    def counting(prompt, model, key, timeout):
        calls.append(prompt)
        return _paper_lover(prompt, model, key, timeout)

    monkeypatch.setitem(ranking.PROVIDERS, "gemini", ("GEMINI_API_KEY", "m", counting))
    run_ranking(seeded)
    for _ in range(2):
        todo = [row_to_dict(r) for r in unscored_items(seeded, "hash1")]
        if todo:
            set_llm_results(seeded, ranking.llm_scores(todo, "p", provider="gemini"),
                            "gemini:m", "hash1")
    assert len(calls) == 1, "the second run had nothing new to score"

    # Editing profile.md changes the hash, and that must re-queue everything.
    assert len(unscored_items(seeded, "hash2")) == 5


def test_heuristic_rescoring_does_not_erase_the_models_judgement(seeded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setitem(ranking.PROVIDERS, "gemini", ("GEMINI_API_KEY", "m", _paper_lover))
    run_ranking(seeded)
    todo = [row_to_dict(r) for r in unscored_items(seeded, "hash1")]
    set_llm_results(seeded, ranking.llm_scores(todo, "p", provider="gemini"), "gemini:m", "hash1")

    run_ranking(seeded)  # tomorrow's run does the free passes again
    paper = [r for r in query_items(seeded) if r["source_id"] == "a1"][0]
    assert paper["score"] == 98.0 and paper["scored_by"] == "gemini:m"
