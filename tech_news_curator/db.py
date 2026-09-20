from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Item
from .normalize import canonical_url, cluster_key

SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    canonical TEXT,
    cluster_key TEXT,
    discussion_url TEXT,
    author TEXT,
    published_ts TEXT,
    summary TEXT,
    points INTEGER NOT NULL DEFAULT 0,
    comments INTEGER NOT NULL DEFAULT 0,
    metric TEXT,
    rank INTEGER NOT NULL DEFAULT 0,
    tags TEXT,
    image TEXT,
    first_seen TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    popularity REAL,
    popularity_reason TEXT,
    category TEXT,
    score REAL,
    score_reason TEXT,
    scored_by TEXT,
    scored_at TEXT,
    profile_hash TEXT,
    UNIQUE(source, source_id)
);
"""

# Applied after migrations, for the same reason as in the sibling project:
# an index can't be created over a column an older database hasn't grown yet.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_ts);
CREATE INDEX IF NOT EXISTS idx_items_score ON items(score);
CREATE INDEX IF NOT EXISTS idx_items_cluster ON items(cluster_key);
"""

MIGRATIONS = {
    "rank": "INTEGER NOT NULL DEFAULT 0",
    "canonical": "TEXT",
    "cluster_key": "TEXT",
    "image": "TEXT",
    "popularity": "REAL",
    "popularity_reason": "TEXT",
    "category": "TEXT",
    "score": "REAL",
    "score_reason": "TEXT",
    "scored_by": "TEXT",
    "scored_at": "TEXT",
    "profile_hash": "TEXT",
}


@contextmanager
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_TABLE)
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(items)")}
        for column, decl in MIGRATIONS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE items ADD COLUMN {column} {decl}")
        conn.executescript(INDEXES)


def upsert_items(db_path, items: list) -> int:
    """Insert or refresh items. Returns how many rows were written.

    Crowd counts are merged with MAX, not overwritten. A story peaks and then
    slides off the front page; a later run that re-reads it at a lower rank
    (or a source that only reports "points in the last hour") must not erase
    the peak, because peak attention is exactly what `popularity` measures.
    """
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    with connect(db_path) as conn:
        for it in items:
            canon = canonical_url(it.url)
            conn.execute(
                """INSERT INTO items
                     (source, source_id, title, url, canonical, cluster_key,
                      discussion_url, author, published_ts, summary, points,
                      comments, metric, rank, tags, image, first_seen, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source, source_id) DO UPDATE SET
                     title=excluded.title,
                     url=excluded.url,
                     canonical=excluded.canonical,
                     cluster_key=excluded.cluster_key,
                     discussion_url=excluded.discussion_url,
                     author=excluded.author,
                     published_ts=COALESCE(excluded.published_ts, items.published_ts),
                     summary=excluded.summary,
                     points=MAX(excluded.points, items.points),
                     comments=MAX(excluded.comments, items.comments),
                     metric=excluded.metric,
                     rank=excluded.rank,
                     tags=excluded.tags,
                     image=COALESCE(NULLIF(excluded.image, ''), items.image),
                     fetched_at=excluded.fetched_at
                """,
                (
                    it.source, it.source_id, it.title, it.url, canon,
                    cluster_key(it.url, it.title), it.discussion_url, it.author,
                    it.published.isoformat() if it.published else None,
                    it.summary, int(it.points or 0), int(it.comments or 0),
                    it.metric, int(it.rank or 0), json.dumps(it.tags or []),
                    it.image, now, now,
                ),
            )
            written += 1
    return written


ORDERS = {
    # NULLs last in every ordering: an unscored row is unknown, not worst.
    "score": "score IS NULL, score DESC, popularity DESC",
    "popularity": "popularity IS NULL, popularity DESC, published_ts DESC",
    "date": "published_ts IS NULL, published_ts DESC",
    "points": "points DESC",
}


def query_items(db_path, order_by: str = "date", since: str = "", source: str = ""):
    order = ORDERS.get(order_by, ORDERS["date"])
    where, params = [], []
    if since:
        # first_seen is the fallback because plenty of sources publish no
        # date at all, and those rows must not silently vanish from a
        # windowed query.
        where.append("COALESCE(published_ts, first_seen) >= ?")
        params.append(since)
    if source:
        where.append("source = ?")
        params.append(source)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    with connect(db_path) as conn:
        return list(conn.execute(f"SELECT * FROM items{clause} ORDER BY {order}", params))


def row_to_dict(row) -> dict:
    d = dict(row)
    for field in ("tags",):
        try:
            d[field] = json.loads(d.get(field) or "[]")
        except (TypeError, ValueError):
            d[field] = []
    return d


def count_items(db_path) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]


def set_popularity(db_path, values: dict) -> int:
    with connect(db_path) as conn:
        for item_id, (pop, reason) in values.items():
            conn.execute(
                "UPDATE items SET popularity=?, popularity_reason=? WHERE id=?",
                (pop, reason, item_id),
            )
    return len(values)


def set_heuristic_scores(db_path, scores: dict) -> int:
    """Write heuristic scores, but never over an LLM score.

    Same rule as the sibling project: the cheap ranker is the floor, not an
    overwrite. An item the model has already read against the profile keeps
    that judgement until the profile changes.
    """
    written = 0
    with connect(db_path) as conn:
        for item_id, (score, reason) in scores.items():
            cur = conn.execute(
                """UPDATE items SET score=?, score_reason=?, scored_by='heuristic'
                   WHERE id=? AND (scored_by IS NULL OR scored_by='heuristic')""",
                (score, reason, item_id),
            )
            written += cur.rowcount
    return written


def set_llm_results(db_path, results: dict, scored_by: str, profile_hash: str) -> int:
    """Write the model's score, one-line reason and category for each item."""
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        for item_id, r in results.items():
            conn.execute(
                """UPDATE items SET score=?, score_reason=?, category=?,
                       scored_by=?, scored_at=?, profile_hash=? WHERE id=?""",
                (r["score"], r["reason"], r.get("category") or None,
                 scored_by, now, profile_hash, item_id),
            )
    return len(results)


def unscored_items(db_path, profile_hash: str, since: str = ""):
    """Items never scored against this exact (profile, model) pair.

    This is what makes a daily LLM run cost pennies: yesterday's 900 items
    are already scored, so only today's new arrivals are sent.
    """
    params = [profile_hash]
    clause = ""
    if since:
        clause = " AND COALESCE(published_ts, first_seen) >= ?"
        params.append(since)
    with connect(db_path) as conn:
        return list(conn.execute(
            f"""SELECT * FROM items
                WHERE (profile_hash IS NULL OR profile_hash != ?){clause}
                ORDER BY popularity IS NULL, popularity DESC""",
            params,
        ))


def set_categories(db_path, categories: dict) -> int:
    """Write heuristic categories, without clobbering the model's."""
    written = 0
    with connect(db_path) as conn:
        for item_id, cat in categories.items():
            cur = conn.execute(
                "UPDATE items SET category=? WHERE id=? AND (category IS NULL OR category='')",
                (cat, item_id),
            )
            written += cur.rowcount
    return written


def prune(db_path, days: int) -> int:
    """Drop items older than `days`, so the cached database stays small.

    News has a shelf life; this is the difference between a database that
    stabilises around a few thousand rows and one that grows forever inside
    the Actions cache.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM items WHERE COALESCE(published_ts, first_seen) < ?", (cutoff,)
        )
        return cur.rowcount
