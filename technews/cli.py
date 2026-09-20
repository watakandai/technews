from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import categorize as categorizing
from . import popularity as popularity_mod
from . import rank as ranking
from .db import (
    init_db, upsert_items, query_items, row_to_dict, count_items,
    set_popularity, set_heuristic_scores, set_llm_results, set_categories,
    unscored_items, prune,
)
from .fetchers.arxiv import ArxivFetcher
from .fetchers.bluesky import BlueskyFetcher
from .fetchers.github import GitHubTrendingFetcher
from .fetchers.hackernews import HackerNewsFetcher
from .fetchers.lobsters import LobstersFetcher
from .fetchers.reddit import RedditFetcher
from .fetchers.rss import RSSFetcher
from .fetchers.twitter import TwitterFetcher

DEFAULT_DB = Path.home() / ".technews" / "news.db"
FEEDS_FILE = Path(__file__).parent / "feeds.json"

# Sources that can go quiet without erroring - a scraper-ish or credentialed
# path where zero results means "something changed", not "slow news day".
FRAGILE_SOURCES = {"reddit", "bluesky", "twitter"}


def load_feeds():
    data = json.loads(FEEDS_FILE.read_text())
    return data.get("feeds", [])


def feed_hints() -> dict:
    return {f["name"]: f.get("hint", "") for f in load_feeds() if f.get("hint")}


def build_fetchers(args):
    """Every source, with its credentials resolved from the environment.

    Credentialed sources are included unconditionally and decide for
    themselves what to do without a key: Reddit falls back to its public
    feed, X reports itself disabled and returns nothing. A fork with no
    secrets at all still produces a full site.
    """
    fetchers = [
        HackerNewsFetcher(days=args.days, min_points=args.hn_min_points),
        LobstersFetcher(),
        RedditFetcher(
            client_id=os.environ.get("REDDIT_CLIENT_ID", "").strip(),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET", "").strip(),
        ),
        GitHubTrendingFetcher(token=os.environ.get("GITHUB_TOKEN", "").strip()),
        ArxivFetcher(),
        BlueskyFetcher(days=min(args.days, 3)),
        TwitterFetcher(bearer_token=os.environ.get("X_BEARER_TOKEN", "").strip()),
    ]
    fetchers += [RSSFetcher(f["name"], f["url"]) for f in load_feeds()]
    return fetchers


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech news curator")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch_parser = sub.add_parser("fetch", help="pull the latest from every source")
    fetch_parser.add_argument(
        "--days", type=int, default=3, help="how far back to ask sources that take a window"
    )
    fetch_parser.add_argument(
        "--hn-min-points", type=int, default=20,
        help="ignore Hacker News stories below this, to keep the new-page noise out",
    )
    fetch_parser.add_argument("--only", default="", help="comma-separated source names")

    rank_parser = sub.add_parser(
        "rank", help="normalize popularity, then score (heuristic always, LLM optionally)"
    )
    rank_parser.add_argument(
        "--llm", action="store_true",
        help="also score and categorize against profile.md (needs the provider's API key)",
    )
    rank_parser.add_argument("--provider", choices=sorted(ranking.PROVIDERS), default="gemini")
    rank_parser.add_argument("--model", default=None, help="override the provider default")
    rank_parser.add_argument("--profile", default=str(ranking.DEFAULT_PROFILE))
    rank_parser.add_argument(
        "--limit", type=int, default=0,
        help="cap how many items go to the LLM this run (0 = all unscored)",
    )
    rank_parser.add_argument("--batch-size", type=int, default=40)
    rank_parser.add_argument(
        "--min-interval", type=float, default=0,
        help="seconds between LLM requests, to stay under a per-minute limit",
    )
    rank_parser.add_argument(
        "--window-days", type=int, default=7,
        help="only send items this recent to the LLM; older ones aren't worth paying for",
    )
    rank_parser.add_argument("--rescore-all", action="store_true")

    export_parser = sub.add_parser("export", help="write the JSON the static page reads")
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--sort", choices=("score", "popularity", "date"), default="score")
    export_parser.add_argument(
        "--days", type=int, default=10, help="how much history to publish"
    )
    export_parser.add_argument(
        "--no-collapse", action="store_true",
        help="keep every copy of a story instead of merging them into one row",
    )

    list_parser = sub.add_parser("list", help="print the ranked list to the terminal")
    list_parser.add_argument("--sort", choices=("score", "popularity", "date"), default="score")
    list_parser.add_argument("--limit", type=int, default=40)
    list_parser.add_argument("--category", default="")

    prune_parser = sub.add_parser("prune", help="drop items older than N days")
    prune_parser.add_argument("--days", type=int, default=45)

    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    init_db(args.db)

    {"fetch": _cmd_fetch, "rank": _cmd_rank, "export": _cmd_export,
     "list": _cmd_list, "prune": _cmd_prune}[args.cmd](args)


def _cmd_fetch(args) -> None:
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    total = 0
    for fetcher in build_fetchers(args):
        if only and fetcher.name not in only:
            continue
        if getattr(fetcher, "enabled", True) is False:
            print(f"{fetcher.name}: disabled (no credentials)")
            continue
        try:
            items = fetcher.fetch()
        except Exception as exc:
            # One dead feed must never cost the other twenty-odd sources.
            print(f"{fetcher.name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        upsert_items(args.db, items)
        total += len(items)
        print(f"{fetcher.name}: {len(items)} items")
        if not items and fetcher.name in FRAGILE_SOURCES:
            print(
                f"  warning: {fetcher.name} returned 0 items - it is rate-limited or "
                "credential-gated, so check before assuming it was a quiet day.",
                file=sys.stderr,
            )
    print(f"fetched {total} items; {count_items(args.db)} in the database")


def _cmd_rank(args) -> None:
    rows = [row_to_dict(r) for r in query_items(args.db)]
    if not rows:
        print("nothing to rank - run `fetch` first")
        return

    # 1. Popularity, normalized per source, then shared across each cluster.
    scores = popularity_mod.popularity_scores(rows)
    scores = popularity_mod.propagate(rows, scores)
    set_popularity(args.db, scores)
    print(f"popularity: scored {len(scores)}/{len(rows)} items "
          f"({len(rows) - len(scores)} have no crowd signal)")

    # 2. Keyword categories for everything, so filters work without a key.
    hints = feed_hints()
    cats = {r["id"]: categorizing.categorize(r, hints) for r in rows}
    print(f"categories: filled {set_categories(args.db, cats)} empty ones")

    # 3. Heuristic relevance, reading the popularity just written.
    rows = [row_to_dict(r) for r in query_items(args.db)]
    heuristic = ranking.heuristic_scores(rows)
    written = set_heuristic_scores(args.db, heuristic)
    kept = len(heuristic) - written
    print(f"heuristic: scored {written} items" + (f" ({kept} keep their LLM score)" if kept else ""))

    if not args.llm:
        print("(pass --llm to also rank against profile.md)")
        return

    try:
        profile = ranking.load_profile(args.profile)
    except FileNotFoundError as exc:
        print(f"llm: SKIPPED ({exc})", file=sys.stderr)
        return

    model = args.model or ranking.PROVIDERS[args.provider][1]
    phash = ranking.profile_hash(profile, model)
    since = (datetime.now(timezone.utc) - timedelta(days=args.window_days)).isoformat()

    if args.rescore_all:
        todo = [row_to_dict(r) for r in query_items(args.db, since=since)]
    else:
        todo = [row_to_dict(r) for r in unscored_items(args.db, phash, since=since)]
    if args.limit:
        # unscored_items returns most-popular-first, so a cap spends the
        # budget on what most people are already reading.
        todo = todo[: args.limit]
    if not todo:
        print(f"llm: nothing to do - everything recent is scored for profile {phash}")
        return

    print(f"llm: sending {len(todo)} items to {args.provider}/{model} "
          f"in batches of {args.batch_size}")

    def progress(offset, size, note):
        print(f"  batch {offset // args.batch_size + 1} ({size} items): {note}")

    try:
        results = ranking.llm_scores(
            todo, profile, provider=args.provider, model=model,
            batch_size=args.batch_size, min_interval=args.min_interval,
            on_progress=progress,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"llm: SKIPPED ({exc})", file=sys.stderr)
        return

    if results:
        set_llm_results(args.db, results, f"{args.provider}:{model}", phash)
    print(f"llm: scored {len(results)}/{len(todo)} items for profile {phash}")


# What the page reads. Everything else (source_id, fetched_at, profile_hash,
# scored_at) is bookkeeping the browser never touches, and this file is
# fetched on every page load.
EXPORT_FIELDS = (
    "id", "source", "title", "url", "discussion_url", "author", "published_ts",
    "summary", "points", "comments", "metric", "tags", "image", "popularity",
    "popularity_reason", "category", "score", "score_reason", "scored_by",
)
SUMMARY_LIMIT = 260


# Fields where zero is a measurement, not an absence. Everything else is
# dropped when empty to keep the payload small, but a story the model scored
# 0 must not be exported as unscored - the page renders those differently
# and "nothing for you" is a real answer.
KEEP_ZERO = {"score", "popularity"}


def _slim(row: dict) -> dict:
    out = {
        k: row[k] for k in EXPORT_FIELDS
        if k in row and (row[k] not in (None, "", [], 0) or (k in KEEP_ZERO and row[k] == 0))
    }
    summary = out.get("summary") or ""
    if len(summary) > SUMMARY_LIMIT:
        out["summary"] = summary[:SUMMARY_LIMIT].rstrip() + "..."
    return out


def _pick_representative(group: list) -> dict:
    """Which copy of a story to actually show.

    Preference order: the highest-scoring one, then one that has a summary
    to read, then one with an image. A publisher's own copy usually wins on
    those tiebreaks over an aggregator's bare link, which is what you want -
    the aggregator's contribution (its score and its comment thread) is
    merged onto the winner rather than lost.
    """
    return max(
        group,
        key=lambda r: (
            r.get("score") or 0,
            bool(r.get("summary")),
            bool(r.get("image")),
            r.get("popularity") or 0,
        ),
    )


def collapse(rows: list) -> list:
    """Merge every copy of the same story into one row carrying all its links."""
    groups = {}
    for row in rows:
        key = row.get("cluster_key") or f"id:{row['id']}"
        groups.setdefault(key, []).append(row)

    merged = []
    for group in groups.values():
        best = dict(_pick_representative(group))
        others = [r for r in group if r["id"] != best["id"]]
        # The best score, popularity and category anywhere in the cluster
        # win: one source having read it properly is enough.
        for row in others:
            if (row.get("score") or 0) > (best.get("score") or 0):
                best["score"], best["score_reason"] = row["score"], row.get("score_reason")
            if (row.get("popularity") or 0) > (best.get("popularity") or 0):
                best["popularity"] = row["popularity"]
                best["popularity_reason"] = row.get("popularity_reason")
            best["category"] = best.get("category") or row.get("category")
            best["image"] = best.get("image") or row.get("image")
            best["summary"] = best.get("summary") or row.get("summary")
        best["also"] = [
            {
                "source": r["source"],
                "url": r.get("discussion_url") or r.get("url") or "",
                "points": r.get("points") or 0,
                "comments": r.get("comments") or 0,
                "metric": r.get("metric") or "",
            }
            for r in sorted(others, key=lambda r: -(r.get("points") or 0))
        ]
        merged.append(best)
    return merged


def _cmd_export(args) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    rows = [row_to_dict(r) for r in query_items(args.db, order_by=args.sort, since=since)]
    total = len(rows)
    if not args.no_collapse:
        rows = collapse(rows)
        # Collapsing reshuffles: a merged row may have inherited a better
        # score than the position it was sorted into.
        rows.sort(key=_sort_key(args.sort), reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": args.days,
        "categories": categorizing.CATEGORIES,
        "items": [dict(_slim(r), also=r.get("also") or []) for r in rows],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One compact item per line: valid JSON, far smaller than indent=2, and
    # still readable as a diff - this file is committed on every daily run,
    # and a single-line blob would make every change unreviewable.
    lines = ",\n".join(
        "  " + json.dumps(item, separators=(",", ":"), default=str)
        for item in payload["items"]
    )
    head = json.dumps(
        {k: v for k, v in payload.items() if k != "items"}, separators=(",", ":"), default=str
    )
    out_path.write_text(f'{head[:-1]},"items":[\n{lines}\n]}}\n' if payload["items"]
                        else head[:-1] + ',"items":[]}\n')
    print(f"exported {len(rows)} rows to {out_path} "
          f"(from {total} raw items over {args.days} days)")


def _sort_key(sort: str):
    if sort == "date":
        return lambda r: (r.get("published_ts") or "")
    field = "popularity" if sort == "popularity" else "score"
    return lambda r: (r.get(field) if r.get(field) is not None else -1)


def _cmd_list(args) -> None:
    rows = [row_to_dict(r) for r in query_items(args.db, order_by=args.sort)]
    rows = collapse(rows)
    rows.sort(key=_sort_key(args.sort), reverse=True)
    if args.category:
        rows = [r for r in rows if (r.get("category") or "") == args.category]
    for row in rows[: args.limit or None]:
        score = f"{row['score']:5.1f}" if row.get("score") is not None else "    -"
        pop = f"{row['popularity']:4.0f}" if row.get("popularity") is not None else "   -"
        cat = (row.get("category") or "other")[:10]
        also = f" +{len(row.get('also') or [])}" if row.get("also") else ""
        print(f"{score} pop{pop} {cat:10} {row['source'][:12]:12} "
              f"{(row['title'] or '')[:64]:64}{also}")


def _cmd_prune(args) -> None:
    removed = prune(args.db, args.days)
    print(f"pruned {removed} items older than {args.days} days; {count_items(args.db)} remain")


if __name__ == "__main__":
    main()
