"""Reddit, by two routes, because the easy one is unreliable.

Anonymous requests from a datacenter IP - which is what a GitHub Actions
runner is - get 429'd by Reddit within a couple of calls. So:

- With REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET set (a free "script" app at
  reddit.com/prefs/apps), this uses the OAuth API: reliable, and it returns
  real upvote and comment counts, which is a genuine popularity signal.
- Without them it falls back to the public .rss feed, which needs no
  credentials but is rate-limited, carries no scores, and only gives the
  ordering. That ordering is still worth having: `rank` feeds popularity.
"""
from __future__ import annotations
import base64
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from ..models import Item
from .base import get_bytes, get_json, post_form
from . import rss

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_URL = "https://oauth.reddit.com/r/{subs}/top?t={window}&limit={limit}"
RSS_URL = "https://www.reddit.com/r/{subs}/top.rss?t={window}&limit={limit}"

DEFAULT_SUBS = (
    "programming", "technology", "MachineLearning", "LocalLLaMA",
    "robotics", "hardware", "compsci", "devops",
)


class RedditFetcher:
    name = "reddit"

    def __init__(self, subs=DEFAULT_SUBS, window: str = "day", limit: int = 100,
                 client_id: str = "", client_secret: str = "", timeout: int = 20):
        self.subs = list(subs)
        self.window = window
        self.limit = limit
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def fetch(self) -> list:
        # One multireddit request instead of eight: Reddit's "+" syntax
        # merges subreddits server-side, and one call is one rate-limit hit.
        subs = "+".join(self.subs)
        if self.authenticated:
            return self._fetch_oauth(subs)
        return self._fetch_rss(subs)

    # -- authenticated -----------------------------------------------------
    def _token(self) -> str:
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        data = post_form(
            TOKEN_URL,
            {"grant_type": "client_credentials"},
            self.timeout,
            {"Authorization": f"Basic {basic}"},
        )
        token = data.get("access_token")
        if not token:
            raise ValueError(f"reddit returned no access_token: {str(data)[:120]}")
        return token

    def _fetch_oauth(self, subs: str) -> list:
        token = self._token()
        url = OAUTH_URL.format(subs=subs, window=self.window, limit=min(self.limit, 100))
        data = get_json(url, self.timeout, {"Authorization": f"bearer {token}"})
        return self.parse(data)

    def parse(self, data: dict) -> list:
        items = []
        for pos, child in enumerate((data.get("data") or {}).get("children") or [], 1):
            post = child.get("data") or {}
            post_id = post.get("id")
            title = (post.get("title") or "").strip()
            if not post_id or not title:
                continue
            permalink = "https://www.reddit.com" + (post.get("permalink") or "")
            # is_self marks a text post: its `url` is the thread itself, so
            # letting it through as an outbound link would cluster every
            # self-post on a subreddit under the same domain.
            outbound = "" if post.get("is_self") else (post.get("url_overridden_by_dest") or post.get("url") or "")
            items.append(
                Item(
                    source=self.name,
                    source_id=post_id,
                    title=title,
                    url=outbound,
                    discussion_url=permalink,
                    author=post.get("author") or "",
                    published=_epoch(post.get("created_utc")),
                    summary=(post.get("selftext") or "")[:600],
                    points=int(post.get("score") or 0),
                    comments=int(post.get("num_comments") or 0),
                    metric="upvotes",
                    rank=pos,
                    tags=[f"r/{post.get('subreddit')}"] if post.get("subreddit") else [],
                    image=_image(post),
                )
            )
        return items

    # -- anonymous fallback ------------------------------------------------
    def _fetch_rss(self, subs: str) -> list:
        """Parse the public Atom feed directly.

        Not RSSFetcher.parse: the outbound article URL only exists as the
        href of the `[link]` anchor inside the rendered <content>, and the
        generic parser strips tags (dropping attributes) before any caller
        sees it. So the raw content is read here instead.
        """
        url = RSS_URL.format(subs=subs, window=self.window, limit=min(self.limit, 100))
        return self.parse_rss(get_bytes(url, self.timeout))

    def parse_rss(self, xml_bytes: bytes) -> list:
        root = ET.fromstring(xml_bytes)
        items = []
        for pos, entry in enumerate(root.findall(f"{{{rss.ATOM}}}entry")[: self.limit], 1):
            title = rss._text(entry, "title")
            thread = rss._link(entry)
            if not title or not thread:
                continue
            body = rss._body(entry)
            items.append(
                Item(
                    source=self.name,
                    source_id=rss._text(entry, "id") or thread,
                    url=_outbound(body) or thread,
                    discussion_url=thread,
                    title=title,
                    author=rss._author(entry),
                    published=rss._date(entry),
                    summary="",  # the feed body is chrome, not a description
                    # "rank" is a claim about the ORDERING being meaningful:
                    # this is the /top feed, so position is a real (if coarse)
                    # popularity signal. A publisher's newest-first feed sets
                    # metric="" instead, and is scored as having no signal.
                    metric="rank",
                    rank=pos,
                    tags=[t for t in rss._categories(entry) if t],
                    image=rss._image(entry, body),
                )
            )
        return items


def _outbound(body: str) -> str:
    """The article URL from the feed's `<a href=...>[link]</a>` anchor.

    A self-post links back to its own thread here, which is correct: there
    is no article, and Item/cluster_key treat it as discussion-only.
    """
    match = re.search(r'href="([^"]+)"[^>]*>\s*\[link\]', body or "")
    return html.unescape(match.group(1)) if match else ""


def _epoch(value):
    if not value:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _image(post: dict) -> str:
    preview = ((post.get("preview") or {}).get("images") or [{}])[0]
    url = (preview.get("source") or {}).get("url") or ""
    return url.replace("&amp;", "&")
