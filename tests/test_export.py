import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from tech_news_curator.cli import _slim, _sort_key, collapse
from tech_news_curator.db import init_db, upsert_items
from tech_news_curator.models import Item


def row(item_id, source, score=None, popularity=None, cluster="u:same", **kw):
    base = {"id": item_id, "source": source, "score": score, "popularity": popularity,
            "cluster_key": cluster, "title": f"{source} copy", "summary": "",
            "image": "", "category": "", "points": 0, "comments": 0, "metric": ""}
    base.update(kw)
    return base


def test_collapse_merges_copies_and_keeps_every_discussion_link():
    rows = [
        row(1, "hackernews", score=40, popularity=95, points=800, comments=300,
            metric="points", discussion_url="https://news.ycombinator.com/item?id=1"),
        row(2, "theverge", score=70, popularity=None, summary="A real writeup",
            image="https://cdn/x.png"),
        row(3, "reddit", score=10, popularity=50, discussion_url="https://reddit.com/r/x"),
    ]
    merged = collapse(rows)
    assert len(merged) == 1
    best = merged[0]
    assert best["source"] == "theverge", "the best-scoring, readable copy represents the story"
    assert best["score"] == 70 and best["popularity"] == 95, \
        "the cluster keeps the best score and the best popularity, wherever each came from"
    assert {a["source"] for a in best["also"]} == {"hackernews", "reddit"}
    assert best["also"][0]["comments"] == 300, "the busiest thread is listed first"


def test_collapse_leaves_distinct_stories_alone():
    rows = [row(1, "hackernews", cluster="u:a"), row(2, "lobsters", cluster="u:b")]
    assert len(collapse(rows)) == 2


def test_rows_with_no_cluster_key_are_never_merged_together():
    rows = [row(1, "bluesky", cluster=""), row(2, "bluesky", cluster="")]
    merged = collapse(rows)
    assert len(merged) == 2, "an empty key means unknown, not 'the same as every other unknown'"


def test_a_cluster_inherits_a_category_and_image_from_any_member():
    rows = [row(1, "hackernews", score=90), row(2, "theverge", score=10,
                category="robotics", image="https://cdn/x.png", summary="text")]
    best = collapse(rows)[0]
    assert best["category"] == "robotics" and best["image"] == "https://cdn/x.png"


def test_slim_drops_empty_fields_and_truncates_long_summaries():
    out = _slim({"id": 1, "source": "hn", "title": "T", "summary": "x" * 400,
                 "author": "", "points": 0, "unknown_field": "dropped"})
    assert "author" not in out and "points" not in out and "unknown_field" not in out
    assert out["summary"].endswith("...") and len(out["summary"]) <= 264


def test_a_zero_score_survives_the_slim_but_a_zero_point_count_does_not():
    out = _slim({"id": 1, "source": "hn", "title": "T", "score": 0.0,
                 "popularity": 0.0, "points": 0})
    assert out["score"] == 0.0 and out["popularity"] == 0.0, \
        "scored-zero is a judgement; the page must not show it as unscored"
    assert "points" not in out


def test_sort_key_puts_unscored_rows_last_not_first():
    rows = [{"score": None}, {"score": 0.0}, {"score": 50.0}]
    ordered = sorted(rows, key=_sort_key("score"), reverse=True)
    assert ordered[-1]["score"] is None, "unknown is not the same claim as zero"


def test_export_writes_line_delimited_json_a_diff_can_show(tmp_path):
    db = tmp_path / "news.db"
    init_db(db)
    now = datetime.now(timezone.utc)
    upsert_items(db, [
        Item(source="hackernews", source_id="1", title="Fresh story",
             url="https://ex.com/a", points=100, published=now),
        Item(source="theverge", source_id="2", title="Same story",
             url="https://ex.com/a?utm_source=x", published=now),
        Item(source="lobsters", source_id="3", title="Ancient story",
             url="https://ex.com/old", published=now - timedelta(days=60)),
    ])
    out = tmp_path / "items.json"
    subprocess.run(
        [sys.executable, "-m", "tech_news_curator.cli", "--db", str(db),
         "export", "--out", str(out), "--days", "10"],
        check=True, capture_output=True,
    )
    text = out.read_text()
    payload = json.loads(text)
    titles = [i["title"] for i in payload["items"]]

    assert len(payload["items"]) == 1, "the two copies collapse; the 60-day-old one is outside the window"
    assert payload["items"][0]["also"][0]["source"] in {"hackernews", "theverge"}
    assert "Ancient story" not in titles
    assert payload["categories"]["robotics"] == "Robotics & Autonomy"
    assert payload["generated_at"]
    # One item per line, so the daily commit is reviewable as a diff.
    assert text.count('\n  {"') == len(payload["items"])
