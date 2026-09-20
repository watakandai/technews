from datetime import datetime, timedelta, timezone

import pytest

from technews.db import (
    count_items, init_db, prune, query_items, row_to_dict, set_heuristic_scores,
    set_llm_results, set_categories, set_popularity, unscored_items, upsert_items,
)
from technews.models import Item


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "news.db"
    init_db(path)
    return path


def make(source="hackernews", source_id="1", title="A story", **kw):
    return Item(source=source, source_id=source_id, title=title, **kw)


def test_init_db_is_idempotent_and_migrates_in_place(db):
    init_db(db)  # a second call must not fail on existing columns or indexes
    assert count_items(db) == 0


def test_upsert_keeps_the_peak_crowd_count(db):
    upsert_items(db, [make(points=300, comments=80)])
    # A later run reads the same story after it slid off the front page.
    upsert_items(db, [make(points=12, comments=90)])
    row = row_to_dict(query_items(db)[0])
    assert row["points"] == 300, "peak attention must survive a lower later reading"
    assert row["comments"] == 90


def test_upsert_stores_the_canonical_url_and_cluster_key(db):
    upsert_items(db, [make(url="https://www.example.com/a/?utm_source=x")])
    row = row_to_dict(query_items(db)[0])
    assert row["canonical"] == "https://example.com/a"
    assert row["cluster_key"] == "u:https://example.com/a"


def test_upsert_does_not_lose_a_date_or_image_to_a_later_empty_one(db):
    when = datetime(2026, 9, 19, tzinfo=timezone.utc)
    upsert_items(db, [make(published=when, image="https://cdn/x.png")])
    upsert_items(db, [make(published=None, image="")])
    row = row_to_dict(query_items(db)[0])
    assert row["published_ts"].startswith("2026-09-19")
    assert row["image"] == "https://cdn/x.png"


def test_heuristic_scores_never_overwrite_an_llm_score(db):
    upsert_items(db, [make()])
    item_id = query_items(db)[0]["id"]
    set_llm_results(db, {item_id: {"score": 90.0, "reason": "great", "category": "robotics"}},
                    "gemini:x", "hash1")
    written = set_heuristic_scores(db, {item_id: (12.0, "meh")})
    row = row_to_dict(query_items(db)[0])
    assert written == 0
    assert row["score"] == 90.0 and row["scored_by"] == "gemini:x"


def test_categories_fill_empties_only(db):
    upsert_items(db, [make(source_id="1"), make(source_id="2")])
    ids = [r["id"] for r in query_items(db)]
    set_llm_results(db, {ids[0]: {"score": 5.0, "reason": "", "category": "robotics"}}, "gemini:x", "h")
    filled = set_categories(db, {ids[0]: "other", ids[1]: "ai_ml"})
    rows = {r["id"]: row_to_dict(r)["category"] for r in query_items(db)}
    assert filled == 1
    assert rows[ids[0]] == "robotics", "the model's category outranks the keyword pass"
    assert rows[ids[1]] == "ai_ml"


def test_unscored_items_excludes_the_current_profile_and_orders_by_popularity(db):
    upsert_items(db, [make(source_id=str(i)) for i in range(3)])
    ids = [r["id"] for r in query_items(db)]
    set_popularity(db, {ids[0]: (10.0, ""), ids[1]: (90.0, ""), ids[2]: (50.0, "")})
    set_llm_results(db, {ids[1]: {"score": 1.0, "reason": "", "category": ""}}, "gemini:x", "hash1")

    todo = [r["id"] for r in unscored_items(db, "hash1")]
    assert ids[1] not in todo, "already scored against this exact profile"
    assert todo == [ids[2], ids[0]], "most popular first, so a --limit spends budget well"

    # Editing profile.md changes the hash, which must re-queue everything.
    assert len(unscored_items(db, "hash2")) == 3


def test_query_window_falls_back_to_first_seen_for_undated_items(db):
    upsert_items(db, [make(source_id="dated", published=datetime(2020, 1, 1, tzinfo=timezone.utc)),
                      make(source_id="undated")])
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    titles = {row_to_dict(r)["source_id"] for r in query_items(db, since=since)}
    assert titles == {"undated"}, "an undated item is current until proven otherwise"


def test_prune_drops_old_rows_and_keeps_recent_ones(db):
    old = datetime.now(timezone.utc) - timedelta(days=90)
    upsert_items(db, [make(source_id="old", published=old), make(source_id="new")])
    assert prune(db, 45) == 1
    assert count_items(db) == 1
