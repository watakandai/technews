"""arXiv, for the research end of the feed.

Aggregators surface papers only once someone writes them up, which is days
late and heavily filtered by what makes a good headline. Going to the source
catches robotics, control, formal methods and optimization work that never
gets a blog post - the categories below are chosen for exactly that.

No popularity signal exists here (arXiv publishes no view or citation count
in the API), so these arrive at points=0 and live or die on the profile
ranker, which is the right call: a preprint's value to one reader has very
little to do with how many other people clicked it.
"""
from __future__ import annotations
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

from ..models import Item
from .base import get_bytes

API = "http://export.arxiv.org/api/query"
ATOM = "http://www.w3.org/2005/Atom"

# cs.RO robotics, cs.AI/cs.LG learning, cs.LO + cs.FL formal methods/logic,
# math.OC optimization and control, eess.SY systems and control.
DEFAULT_CATEGORIES = (
    "cs.RO", "cs.AI", "cs.LG", "cs.LO", "cs.FL", "math.OC", "eess.SY",
)


class ArxivFetcher:
    name = "arxiv"

    def __init__(self, categories=DEFAULT_CATEGORIES, limit: int = 60, timeout: int = 25):
        self.categories = list(categories)
        self.limit = limit
        self.timeout = timeout

    def fetch(self) -> list:
        query = " OR ".join(f"cat:{c}" for c in self.categories)
        url = (
            f"{API}?search_query={urllib.parse.quote(query)}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results={self.limit}"
        )
        return self.parse(get_bytes(url, self.timeout))

    def parse(self, xml_bytes: bytes) -> list:
        root = ET.fromstring(xml_bytes)
        items = []
        for pos, entry in enumerate(root.findall(f"{{{ATOM}}}entry"), 1):
            arxiv_id = _text(entry, "id")
            title = _flat(_text(entry, "title"))
            if not arxiv_id or not title:
                continue
            authors = [
                _flat(a.findtext(f"{{{ATOM}}}name") or "")
                for a in entry.findall(f"{{{ATOM}}}author")
            ]
            cats = [
                c.get("term") for c in entry.findall(f"{{{ATOM}}}category") if c.get("term")
            ]
            items.append(
                Item(
                    source=self.name,
                    # The abs URL carries a version suffix (v1, v2); strip it
                    # so a revision updates the row instead of duplicating it.
                    source_id=re.sub(r"v\d+$", "", arxiv_id.rsplit("/", 1)[-1]),
                    title=title,
                    url=arxiv_id,
                    author=", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                    published=_iso(_text(entry, "published")),
                    summary=_flat(_text(entry, "summary"))[:600],
                    metric="",
                    rank=pos,
                    tags=cats[:6],
                )
            )
        return items


def _text(entry, tag: str) -> str:
    return (entry.findtext(f"{{{ATOM}}}{tag}") or "").strip()


def _flat(text: str) -> str:
    # arXiv hard-wraps titles and abstracts at ~80 columns.
    return re.sub(r"\s+", " ", text or "").strip()


def _iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
