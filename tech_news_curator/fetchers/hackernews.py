from __future__ import annotations
from datetime import datetime, timedelta, timezone

from ..models import Item
from .base import get_json

# Algolia's HN index rather than the official Firebase API. Firebase needs
# one request per story (500+ round trips for a front page plus its tail);
# Algolia answers the same question in one, and returns points and comment
# counts inline, which is exactly the signal this project ranks on.
SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM = "https://news.ycombinator.com/item?id={}"


class HackerNewsFetcher:
    """Top Hacker News stories from the last few days.

    An empty query with a `created_at_i` filter makes Algolia fall back to
    the index's own ranking, which for HN is points-first - so this is
    "the highest-scoring stories in the window", not a keyword search.
    """

    name = "hackernews"

    def __init__(self, days: int = 3, min_points: int = 20, limit: int = 150, timeout: int = 20):
        self.days = days
        self.min_points = min_points
        self.limit = limit
        self.timeout = timeout

    def fetch(self) -> list:
        since = int((datetime.now(timezone.utc) - timedelta(days=self.days)).timestamp())
        filters = f"created_at_i>{since},points>{self.min_points}"
        url = (
            f"{SEARCH_URL}?tags=story&numericFilters={filters}"
            f"&hitsPerPage={min(self.limit, 1000)}"
        )
        return self.parse(get_json(url, self.timeout))

    def parse(self, data: dict) -> list:
        items = []
        for pos, hit in enumerate(data.get("hits") or [], 1):
            hn_id = str(hit.get("objectID") or "").strip()
            title = (hit.get("title") or hit.get("story_title") or "").strip()
            if not hn_id or not title:
                continue
            discussion = HN_ITEM.format(hn_id)
            items.append(
                Item(
                    source=self.name,
                    source_id=hn_id,
                    title=title,
                    # A text post ("Ask HN", "Show HN" with no link) has no
                    # outbound URL; Item falls back to the thread itself.
                    url=(hit.get("url") or "").strip(),
                    discussion_url=discussion,
                    author=hit.get("author") or "",
                    published=_ts(hit.get("created_at_i")),
                    summary=_clean(hit.get("story_text") or hit.get("comment_text") or ""),
                    points=int(hit.get("points") or 0),
                    comments=int(hit.get("num_comments") or 0),
                    metric="points",
                    rank=pos,
                    tags=[t for t in (hit.get("_tags") or []) if _useful_tag(t)],
                )
            )
        return items


def _useful_tag(tag: str) -> bool:
    # _tags is mostly bookkeeping ("story", "author_pg", "story_49764791").
    return tag in ("show_hn", "ask_hn", "front_page", "launch_hn")


def _ts(epoch):
    if not epoch:
        return None
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _clean(html: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", " ", html or "")
    import html as html_mod
    return re.sub(r"\s+", " ", html_mod.unescape(text)).strip()[:600]
