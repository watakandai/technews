from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    """One thing worth reading, from any source.

    Deliberately flat and source-agnostic: a Hacker News submission, a GitHub
    repo trending today, an arXiv preprint and a post on Bluesky all land in
    the same shape, because everything downstream - dedupe, popularity,
    ranking, the page - works on the union rather than on per-source special
    cases.
    """

    source: str
    source_id: str  # stable dedupe key within a source (id, permalink, guid)
    title: str
    # Where the story actually lives. For a link aggregator this is the
    # outbound article, NOT the comment thread - that is what makes the same
    # story submitted to HN, Reddit and Lobsters collapse into one cluster.
    url: str = ""
    # The conversation about it, when the source has one separate from the
    # link. Kept because the comments are often the reason to click.
    discussion_url: str = ""
    author: str = ""
    published: datetime | None = None
    summary: str = ""
    # Raw crowd signal, in whatever unit the source counts in. Never compared
    # across sources directly - `popularity` normalizes these per source.
    points: int = 0
    comments: int = 0
    metric: str = "points"  # what `points` counts: points/upvotes/stars/likes
    # 1-based position in an already-ordered feed, for sources that rank
    # their output but publish no number behind it (Reddit's RSS, a
    # newsletter's running order). 0 means "this source has no ordering".
    # `popularity` falls back to this when `points` is absent, so an
    # order-only source still sorts sensibly instead of flatlining at zero.
    rank: int = 0
    tags: list[str] = field(default_factory=list)
    image: str = ""

    def __post_init__(self) -> None:
        # A source with no separate discussion page (a blog's RSS, arXiv)
        # leaves discussion_url empty; a source that is ONLY a discussion
        # (an Ask HN post, a Bluesky thread) leaves url empty. Normalizing
        # here means the page always has something to link to.
        if not self.url:
            self.url = self.discussion_url
