from __future__ import annotations
from datetime import datetime, timedelta, timezone

from ..models import Item
from .base import get_json

# GitHub has no trending API - the trending page is HTML with no contract.
# The search API is official and stable, and "created recently, sorted by
# stars" is a sharper signal anyway: it surfaces things that got popular
# THIS week rather than long-established repos that merely got a commit.
SEARCH = "https://api.github.com/search/repositories"


class GitHubTrendingFetcher:
    name = "github"

    def __init__(self, days: int = 14, min_stars: int = 50, limit: int = 50,
                 token: str = "", timeout: int = 20):
        self.days = days
        self.min_stars = min_stars
        self.limit = limit
        self.token = token
        self.timeout = timeout

    def fetch(self) -> list:
        since = (datetime.now(timezone.utc) - timedelta(days=self.days)).date().isoformat()
        query = f"created:>{since} stars:>={self.min_stars}"
        url = (
            f"{SEARCH}?q={query.replace(' ', '+').replace('>', '%3E').replace(':', '%3A')}"
            f"&sort=stars&order=desc&per_page={min(self.limit, 100)}"
        )
        headers = {"Accept": "application/vnd.github+json"}
        # Unauthenticated search is 10 requests/minute; inside Actions the
        # automatic GITHUB_TOKEN lifts that without any setup by the user.
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return self.parse(get_json(url, self.timeout, headers))

    def parse(self, data: dict) -> list:
        items = []
        for pos, repo in enumerate(data.get("items") or [], 1):
            full_name = repo.get("full_name") or ""
            if not full_name:
                continue
            language = repo.get("language")
            items.append(
                Item(
                    source=self.name,
                    source_id=str(repo.get("id") or full_name),
                    # The bare repo name means nothing out of context, and
                    # this title is what the ranker and the reader both see.
                    title=f"{full_name} - {(repo.get('description') or 'no description').strip()}"[:200],
                    url=repo.get("html_url") or "",
                    author=(repo.get("owner") or {}).get("login") or "",
                    published=_iso(repo.get("created_at")),
                    summary=(repo.get("description") or "")[:600],
                    points=int(repo.get("stargazers_count") or 0),
                    comments=int(repo.get("open_issues_count") or 0),
                    metric="stars",
                    rank=pos,
                    tags=([language] if language else []) + list(repo.get("topics") or [])[:6],
                )
            )
        return items


def _iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
