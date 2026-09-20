"""Bluesky, as the microblog source that actually works without paying.

X's v2 search endpoint is not on its free tier, so a project that has to run
unattended in CI can't lean on it (see twitter.py, which supports it for
anyone who does have a key). Bluesky's AppView answers the same question -
what are people in tech posting and boosting right now - over a public,
unauthenticated, documented API, and carries like/repost/reply counts, which
is a real popularity signal rather than a guess.
"""
from __future__ import annotations
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from ..models import Item
from .base import USER_AGENT, get_json

# public.api.bsky.app is the documented read-only AppView; api.bsky.app
# serves the same routes and is tried second because the first is behind a
# CDN that sometimes 403s a datacenter IP outright.
HOSTS = ("https://public.api.bsky.app", "https://api.bsky.app")
SEARCH = "/xrpc/app.bsky.feed.searchPosts"
POST_URL = "https://bsky.app/profile/{handle}/post/{rkey}"

DEFAULT_QUERIES = (
    "LLM", "AI agents", "robotics", "open source", "chip",
    "developer tools", "self-driving", "machine learning",
)


class BlueskyFetcher:
    name = "bluesky"

    def __init__(self, queries=DEFAULT_QUERIES, days: int = 2, limit: int = 25,
                 min_likes: int = 25, require_link: bool = True, timeout: int = 20):
        self.queries = list(queries)
        self.days = days
        self.limit = limit
        # Keyword search on a social network returns overwhelmingly opinion
        # about a topic rather than news of it - "top LLM posts" is mostly
        # jokes. Requiring an outbound link is the cheap, high-precision
        # filter: a post carrying a URL is usually passing something along.
        # It also gives the post a cluster key shared with the HN and Reddit
        # submissions of the same link, which is the point of having it here.
        self.require_link = require_link
        # Cap the redirect-following: it is one extra request per item.
        self.resolve_limit = 40
        # Without a floor this is a firehose of one-like posts. The bar is
        # what turns "someone said something" into "people are talking".
        self.min_likes = min_likes
        self.timeout = timeout

    def fetch(self) -> list:
        since = (datetime.now(timezone.utc) - timedelta(days=self.days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        seen, items = set(), []
        for query in self.queries:
            params = urllib.parse.urlencode(
                {"q": query, "sort": "top", "since": since, "limit": min(self.limit, 100)}
            )
            try:
                data = self._get(f"{SEARCH}?{params}")
            except (urllib.error.URLError, ValueError):
                # One dud query shouldn't cost the other seven.
                continue
            for item in self.parse(data, query):
                if item.source_id in seen:
                    continue  # the same post matches several queries
                seen.add(item.source_id)
                items.append(item)
        for item in items[:self.resolve_limit]:
            item.url = resolve_redirect(item.url)
        return items

    def _get(self, path: str):
        last = None
        for host in HOSTS:
            try:
                return get_json(host + path, self.timeout)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (403, 429, 502, 503):
                    raise
        raise last

    def parse(self, data: dict, query: str = "") -> list:
        items = []
        for pos, post in enumerate(data.get("posts") or [], 1):
            uri = post.get("uri") or ""
            record = post.get("record") or {}
            text = (record.get("text") or "").strip()
            likes = int(post.get("likeCount") or 0)
            if not uri or not text or likes < self.min_likes:
                continue
            external = _external(post, record)
            if self.require_link and not external:
                continue
            handle = (post.get("author") or {}).get("handle") or ""
            rkey = uri.rsplit("/", 1)[-1]
            items.append(
                Item(
                    source=self.name,
                    source_id=uri,
                    # A post has no headline, so its first line becomes one.
                    title=_headline(text, external),
                    url=external,
                    discussion_url=POST_URL.format(handle=handle, rkey=rkey) if handle else "",
                    author=handle,
                    published=_iso(record.get("createdAt") or post.get("indexedAt")),
                    summary=text[:600],
                    # Reposting costs more than liking, so it says more; the
                    # weighting is what keeps a merely-likeable joke below a
                    # link the field actually passed around.
                    points=likes + 2 * int(post.get("repostCount") or 0),
                    comments=int(post.get("replyCount") or 0),
                    metric="likes+reposts",
                    rank=pos,
                    tags=[query] if query else [],
                )
            )
        return items


def _headline(text: str, url: str = "") -> str:
    """A post's first sentence, as a headline.

    Falls back to the link's own slug when the post has nothing usable to
    say - plenty of posts are a bare URL, or "this", and a row titled
    "2 days" tells a reader nothing and gives the ranker nothing to read.
    """
    first = re.split(r"(?<=[.!?])\s|\n", text.strip(), 1)[0]
    first = re.sub(r"https?://\S+", "", first)
    first = re.sub(r"\s+", " ", first).strip(" -\u2013\u2014:")
    if len(first) < 20 and url:
        slug = _slug_title(url)
        if slug:
            first = f"{slug} ({first})" if first else slug
    return (first[:157] + "...") if len(first) > 160 else (first or text[:160])


def _slug_title(url: str) -> str:
    """Turn ".../blog/benchmarking-gpu-kernels" into "Benchmarking gpu kernels"."""
    path = urllib.parse.urlsplit(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug)
    slug = re.sub(r"^\d{4,}[-_]?", "", slug)  # leading article ids
    words = [w for w in re.split(r"[-_]+", slug) if w and not w.isdigit()]
    return " ".join(words).strip().capitalize() if len(words) >= 2 else ""


# Link shorteners, which would otherwise give the same article a different
# identity per source and defeat cross-source agreement entirely. Resolved
# with a HEAD request, capped per run, and skipped silently on failure.
SHORTENERS = {
    "trib.al", "ft.trib.al", "t.co", "bit.ly", "buff.ly", "ow.ly", "dlvr.it",
    "tinyurl.com", "nyti.ms", "wapo.st", "reut.rs", "on.ft.com", "apple.co",
    "econ.st", "cnb.cx", "bloom.bg", "engt.co", "zd.net",
}


def resolve_redirect(url: str, timeout: int = 8) -> str:
    """Follow a shortener to its destination. Returns the input on any failure."""
    host = urllib.parse.urlsplit(url).netloc.lower()
    if host.lstrip("www.") not in SHORTENERS and host not in SHORTENERS:
        return url
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.geturl() or url
    except Exception:
        return url


def _external(post: dict, record: dict) -> str:
    """The link a post is pointing at, if any.

    Preferred over the post itself: a post linking to an announcement is the
    same story as the announcement, and only a shared URL makes it cluster
    with the HN and Reddit copies of it.
    """
    for embed in (post.get("embed") or {}, record.get("embed") or {}):
        external = embed.get("external") or {}
        if external.get("uri"):
            return external["uri"]
        if (embed.get("$type") or "").startswith("app.bsky.embed.external"):
            uri = (embed.get("external") or {}).get("uri")
            if uri:
                return uri
    for facet in record.get("facets") or []:
        for feature in facet.get("features") or []:
            if feature.get("uri"):
                return feature["uri"]
    return ""


def _iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
