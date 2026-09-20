from __future__ import annotations
from datetime import datetime

from ..models import Item
from .base import get_json

HOTTEST = "https://lobste.rs/hottest.json"


class LobstersFetcher:
    """Lobsters' hottest page.

    Small (25 stories) and heavily moderated, which makes it a useful
    counterweight: what appears here and on Hacker News at once is a much
    better "this matters" signal than a high score on either alone.
    """

    name = "lobsters"

    def __init__(self, url: str = HOTTEST, timeout: int = 20):
        self.url = url
        self.timeout = timeout

    def fetch(self) -> list:
        return self.parse(get_json(self.url, self.timeout))

    def parse(self, data: list) -> list:
        items = []
        for pos, s in enumerate(data or [], 1):
            short_id = (s.get("short_id") or "").strip()
            title = (s.get("title") or "").strip()
            if not short_id or not title:
                continue
            items.append(
                Item(
                    source=self.name,
                    source_id=short_id,
                    title=title,
                    url=(s.get("url") or "").strip(),
                    discussion_url=s.get("comments_url") or "",
                    author=((s.get("submitter_user") or {}).get("username")
                            if isinstance(s.get("submitter_user"), dict)
                            else s.get("submitter_user")) or "",
                    published=_iso(s.get("created_at")),
                    summary=(s.get("description_plain") or "")[:600],
                    points=int(s.get("score") or 0),
                    comments=int(s.get("comment_count") or 0),
                    metric="points",
                    rank=pos,
                    tags=list(s.get("tags") or []),
                )
            )
        return items


def _iso(value):
    if not value:
        return None
    try:
        # Lobsters sends an offset ("-05:00"); fromisoformat handles that,
        # but not the "Z" some other sources use.
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
