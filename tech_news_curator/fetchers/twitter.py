"""X (Twitter), for anyone who has a key for it.

This is opt-in and off by default, and that is a limitation of X rather than
a choice: the v2 recent-search endpoint used here is not on the free tier,
so an unattended daily job cannot rely on it. Set X_BEARER_TOKEN and it
joins the rotation; leave it unset and the fetcher reports itself disabled
and is skipped, with bluesky.py covering the same ground for free.
"""
from __future__ import annotations
import urllib.parse
from datetime import datetime, timedelta, timezone

from ..models import Item
from .base import get_json

SEARCH = "https://api.x.com/2/tweets/search/recent"
TWEET_URL = "https://x.com/{user}/status/{id}"

# Accounts-and-topics rather than bare keywords: a bare "AI" query on X is
# unusable. -is:retweet keeps the original post rather than its 400 copies.
DEFAULT_QUERY = (
    "(robotics OR \"formal methods\" OR \"reinforcement learning\" OR LLM OR "
    "\"open source\" OR chips OR funding) "
    "(url:arxiv.org OR url:github.com OR has:links) -is:retweet -is:reply lang:en"
)


class TwitterFetcher:
    name = "twitter"

    def __init__(self, bearer_token: str = "", query: str = DEFAULT_QUERY,
                 hours: int = 36, limit: int = 100, min_likes: int = 25,
                 timeout: int = 20):
        self.bearer_token = bearer_token
        self.query = query
        self.hours = hours
        self.limit = limit
        self.min_likes = min_likes
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.bearer_token)

    def fetch(self) -> list:
        if not self.enabled:
            return []
        # X requires start_time to be at least 10 seconds in the past and no
        # more than 7 days back on the recent-search endpoint.
        start = (datetime.now(timezone.utc) - timedelta(hours=self.hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        params = urllib.parse.urlencode({
            "query": self.query,
            "start_time": start,
            "max_results": min(max(self.limit, 10), 100),
            "tweet.fields": "created_at,public_metrics,entities,author_id,lang",
            "expansions": "author_id",
            "user.fields": "username,name",
        })
        data = get_json(
            f"{SEARCH}?{params}", self.timeout,
            {"Authorization": f"Bearer {self.bearer_token}"},
        )
        return self.parse(data)

    def parse(self, data: dict) -> list:
        users = {
            u["id"]: u.get("username", "")
            for u in ((data.get("includes") or {}).get("users") or [])
            if u.get("id")
        }
        items = []
        for pos, tweet in enumerate(data.get("data") or [], 1):
            metrics = tweet.get("public_metrics") or {}
            likes = int(metrics.get("like_count") or 0)
            text = (tweet.get("text") or "").strip()
            if not text or likes < self.min_likes:
                continue
            username = users.get(tweet.get("author_id"), "")
            tweet_id = tweet.get("id") or ""
            items.append(
                Item(
                    source=self.name,
                    source_id=tweet_id,
                    title=_headline(text),
                    url=_external(tweet),
                    discussion_url=TWEET_URL.format(user=username or "i", id=tweet_id),
                    author=username,
                    published=_iso(tweet.get("created_at")),
                    summary=text[:600],
                    points=likes + 2 * int(metrics.get("retweet_count") or 0),
                    comments=int(metrics.get("reply_count") or 0),
                    metric="likes+reposts",
                    rank=pos,
                )
            )
        return items


def _external(tweet: dict) -> str:
    """The first non-Twitter link, expanded past t.co."""
    for url in ((tweet.get("entities") or {}).get("urls") or []):
        expanded = url.get("expanded_url") or url.get("url") or ""
        if expanded and "twitter.com" not in expanded and "//x.com" not in expanded:
            return expanded
    return ""


def _headline(text: str) -> str:
    import re
    first = re.split(r"(?<=[.!?])\s|\n", text.strip(), 1)[0]
    first = re.sub(r"https?://\S+", "", first)
    first = re.sub(r"\s+", " ", first).strip(" -:")
    return (first[:157] + "...") if len(first) > 160 else (first or text[:160])


def _iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
