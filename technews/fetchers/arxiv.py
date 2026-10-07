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
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

from ..models import Item
from .base import get_bytes

API = "http://export.arxiv.org/api/query"
ATOM = "http://www.w3.org/2005/Atom"

# cs.RO robotics, cs.AI/cs.LG learning, cs.LO + cs.FL formal methods/logic,
# math.OC optimization and control, eess.SY systems and control, cs.MA
# multi-agent systems. These map onto the robot_* categories one for one,
# which is the point: arXiv is the only source that files its own subject.
DEFAULT_CATEGORIES = (
    "cs.RO", "cs.AI", "cs.LG", "cs.LO", "cs.FL", "math.OC", "eess.SY", "cs.MA",
)

# Robot foundation models get a query of their own. In the date-sorted
# listing above, cs.LG and cs.AI publish several times what cs.RO does, so
# 100 slots hold only a handful of robotics papers, and the VLA, world-model
# and end-to-end driving work the reader cares most about is easily crowded
# out. Same source name, same ids: a paper both queries return is one row.
# Multi-agent planning rides in the same request, because arXiv wants one
# call every few seconds and its papers sit mostly in cs.MA and cs.AI, which
# the general listing above shares with all of machine learning.
ROBOT_LEARNING_QUERY = (
    "((cat:cs.RO OR cat:cs.CV) AND ("
    'abs:"vision-language-action" OR abs:"robot foundation model" OR '
    'abs:"generalist policy" OR abs:"generalist robot" OR abs:"diffusion policy" OR '
    'abs:"world model" OR abs:"end-to-end driving" OR abs:"end-to-end autonomous driving" OR '
    'abs:"cross-embodiment" OR abs:"imitation learning" OR abs:"flow matching policy"))'
    " OR ((cat:cs.RO OR cat:cs.MA OR cat:cs.AI) AND ("
    'abs:"multi-robot" OR abs:"multi-agent path finding" OR abs:"multi-agent pathfinding" OR '
    'abs:"multi-agent planning" OR abs:"multi-agent reinforcement learning" OR '
    'abs:"task allocation" OR abs:"swarm"))'
)


# The listing is generated per request and a 100-entry one has taken over
# 25 seconds from CI, hence the long default timeout.
# arXiv asks API clients for three seconds between calls and answers 429 or
# 503 to a client that doesn't wait. Two queries now run back to back.
PAUSE = 3.5
_last_call = [float("-inf")]   # monotonic time can start near zero


class ArxivFetcher:
    name = "arxiv"

    def __init__(self, categories=DEFAULT_CATEGORIES, limit: int = 100, timeout: int = 60,
                 query: str = ""):
        self.categories = list(categories)
        self.limit = limit
        self.timeout = timeout
        # A raw arXiv search_query; when set it replaces the category list.
        self.query = query

    def fetch(self) -> list:
        query = self.query or " OR ".join(f"cat:{c}" for c in self.categories)
        url = (
            f"{API}?search_query={urllib.parse.quote(query)}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results={self.limit}"
        )
        wait = _last_call[0] + PAUSE - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return self.parse(get_bytes(url, self.timeout))
        finally:
            _last_call[0] = time.monotonic()

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
