"""One parser for RSS 2.0 and Atom, used by every feed-shaped source.

The long tail of "tech news from everywhere" is feeds: publications, company
engineering blogs, newsletters, individual writers. They have no scores and
no API, and a dedicated fetcher per outlet would be a hundred near-identical
files - so outlets live as data in sources.json and share this parser.

Feeds carry no popularity signal at all. That is not a gap to paper over
with a fabricated number: these items enter with points=0 and are ranked on
cross-source agreement (did an aggregator pick this up?) and on the LLM's
read of the profile, which is the honest answer to "should I read this?"
"""
from __future__ import annotations
import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

from ..models import Item
from .base import get_bytes

ATOM = "http://www.w3.org/2005/Atom"
MEDIA = "http://search.yahoo.com/mrss/"
DC = "http://purl.org/dc/elements/1.1/"
CONTENT = "http://purl.org/rss/1.0/modules/content/"
NS = {"a": ATOM, "media": MEDIA, "dc": DC, "content": CONTENT}

TAG_RE = re.compile(r"<[^>]+>")
IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)', re.I)


class RSSFetcher:
    """A single feed. `name` is the source id the rest of the app groups by."""

    def __init__(self, name: str, url: str, tags=None, limit: int = 40, timeout: int = 20):
        self.name = name
        self.url = url
        self.tags = list(tags or [])
        self.limit = limit
        self.timeout = timeout

    def fetch(self) -> list:
        return self.parse(get_bytes(self.url, self.timeout))

    def parse(self, xml_bytes: bytes) -> list:
        root = ET.fromstring(xml_bytes)
        entries = root.findall("./channel/item") or root.findall(f"{{{ATOM}}}entry")
        items = []
        for pos, entry in enumerate(entries[: self.limit], 1):
            title = _text(entry, "title")
            link = _link(entry)
            if not title or not link:
                continue
            body = _body(entry)
            items.append(
                Item(
                    source=self.name,
                    # Prefer the feed's own id: some publishers rewrite URLs
                    # (adding a date path, moving to a new CMS) and a
                    # URL-keyed row would re-enter as a duplicate.
                    source_id=_text(entry, "guid") or _text(entry, "id") or link,
                    title=title,
                    url=link,
                    author=_author(entry),
                    published=_date(entry),
                    summary=_strip(body)[:600],
                    metric="",  # no crowd signal exists for a feed
                    rank=pos,
                    tags=self.tags + _categories(entry),
                    image=_image(entry, body),
                )
            )
        return items


def _find(entry, tag: str):
    return entry.find(tag) if entry.find(tag) is not None else entry.find(f"{{{ATOM}}}{tag}")


def _text(entry, tag: str) -> str:
    el = _find(entry, tag)
    if el is None:
        return ""
    return html.unescape("".join(el.itertext())).strip()


def _link(entry) -> str:
    el = entry.find("link")
    if el is not None and (el.text or "").strip():
        return el.text.strip()
    # Atom puts the URL in an attribute, and may list several: the
    # alternate is the human page, the others are replies/enclosures.
    links = entry.findall(f"{{{ATOM}}}link")
    for want in ("alternate", None):
        for link in links:
            if link.get("rel", "alternate") == (want or link.get("rel", "alternate")):
                if link.get("href"):
                    return link.get("href").strip()
    return ""


def _body(entry) -> str:
    for tag in (f"{{{CONTENT}}}encoded", "description", "summary", "content"):
        el = entry.find(tag) if tag.startswith("{") else _find(entry, tag)
        if el is not None:
            text = html.unescape("".join(el.itertext())).strip()
            if text:
                return text
    return ""


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text or "")).strip()


def _author(entry) -> str:
    el = entry.find(f"{{{DC}}}creator")
    if el is not None and el.text:
        return el.text.strip()
    el = entry.find(f"{{{ATOM}}}author/{{{ATOM}}}name") or _find(entry, "author")
    if el is not None:
        return html.unescape("".join(el.itertext())).strip()[:80]
    return ""


DATE_TAGS = ("pubDate", f"{{{DC}}}date", f"{{{ATOM}}}published", f"{{{ATOM}}}updated", "updated")


def _date(entry):
    for tag in DATE_TAGS:
        el = entry.find(tag)
        if el is None or not (el.text or "").strip():
            continue
        raw = el.text.strip()
        # RSS uses RFC 822 ("Fri, 19 Sep 2026 17:00:00 GMT"), Atom uses
        # ISO 8601. Feeds in the wild mix them, so try both on every tag.
        try:
            return parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def _categories(entry) -> list:
    out = []
    for el in entry.findall("category") + entry.findall(f"{{{ATOM}}}category"):
        value = (el.get("term") or el.text or "").strip()
        if value and value.lower() not in {c.lower() for c in out}:
            out.append(value[:30])
    return out[:6]


def _image(entry, body: str) -> str:
    for el in entry.findall(f"{{{MEDIA}}}content") + entry.findall(f"{{{MEDIA}}}thumbnail"):
        if (el.get("url") or "").strip():
            return el.get("url").strip()
    for el in entry.findall("enclosure"):
        if (el.get("type") or "").startswith("image") and el.get("url"):
            return el.get("url").strip()
    match = IMG_RE.search(body or "")
    return match.group(1) if match else ""


def now_utc():
    return datetime.now(timezone.utc)
