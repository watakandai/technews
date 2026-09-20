"""Deciding when two listings are the same story.

This is the load-bearing module for "curates tech news from everywhere": the
whole point of pulling ten sources is that agreement between them is evidence,
and agreement can only be counted once identical stories stop looking like
ten separate ones.

Two mechanisms, in order of trust:

- `canonical_url` - the same article linked from HN, Reddit, Lobsters and a
  newsletter is the same URL wearing different tracking parameters. This is
  exact and cheap, and it catches most of it.
- `title_key` - for sources that don't share a link at all (a Bluesky post
  about a launch, a press release republished on two domains). Fuzzier, so
  it is only consulted when there is no URL to compare.
"""
from __future__ import annotations
import re
import urllib.parse

# Campaign/analytics junk. Dropping these is safe: no site serves different
# content based on them, and leaving them in splits one story into several.
TRACKING_PREFIXES = ("utm_", "mc_", "pk_", "hsa_", "at_", "oly_")
TRACKING_EXACT = {
    "ref", "ref_src", "ref_url", "source", "cmpid", "cmp", "fbclid", "gclid",
    "igshid", "mkt_tok", "spm", "sh", "share_id", "guccounter", "smid",
    "__twitter_impression", "s", "t", "si", "feature", "guce_referrer",
}
# Query parameters that genuinely select content and must be kept, even
# though they look generic. Dropping `v` would collapse all of YouTube into
# one story, and `id` does the same on a dozen CMSes.
MEANINGFUL = {"v", "id", "p", "story", "article", "page", "q", "paper", "abs"}

# Hosts where the path alone is the identity and the subdomain is noise.
WWW_RE = re.compile(r"^(www|m|mobile|amp)\.", re.I)
AMP_SUFFIX_RE = re.compile(r"/amp(/|$)|\.amp(/|$)", re.I)

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "with", "at", "by", "from", "as", "it", "its", "that",
    "this", "how", "why", "what", "show", "hn", "ask", "new", "using", "via",
}


def canonical_url(url: str) -> str:
    """A URL reduced to what identifies the document.

    Returns "" for anything unusable, which callers read as "no URL identity
    available, fall back to the title".
    """
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https", ""):
        return ""
    host = WWW_RE.sub("", parts.netloc.lower()).rstrip(".")
    if not host:
        return ""
    # Default ports say nothing about identity.
    host = re.sub(r":(80|443)$", "", host)

    path = AMP_SUFFIX_RE.sub("/", parts.path or "/")
    path = re.sub(r"^/amp(?=/)", "", path)  # cnbc.com/amp/2026/... -> cnbc.com/2026/...
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")

    kept = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=False)
        if _keep_param(k)
    ]
    query = urllib.parse.urlencode(sorted(kept))
    # The fragment is dropped: #section-3 is a place in a document, not a
    # different document. (Hash-routed SPAs are not what news links are.)
    return urllib.parse.urlunsplit(("https", host, path, query, ""))


def _keep_param(key: str) -> bool:
    k = key.lower()
    if k in MEANINGFUL:
        return True
    if k in TRACKING_EXACT:
        return False
    return not any(k.startswith(p) for p in TRACKING_PREFIXES)


def title_tokens(title: str) -> list[str]:
    """Content words of a headline, lowercased, stopwords and markup dropped."""
    text = (title or "").lower()
    # Aggregators bolt a topic prefix onto the headline: "Show HN: ...",
    # "[D] ...", "(2019)". They are about the venue, not the story.
    text = re.sub(r"^\s*(show|ask|tell)\s+hn\s*:", " ", text)
    text = re.sub(r"\[[^\]]{1,12}\]", " ", text)
    text = re.sub(r"\(\s*(19|20)\d{2}\s*\)", " ", text)
    return [w for w in re.findall(r"[a-z0-9][a-z0-9+#.-]*", text) if w not in STOPWORDS]


def title_key(title: str) -> str:
    """A headline reduced to its five most distinctive words, in order.

    Five rather than the whole thing: outlets rewrite the tail of a headline
    ("...", say researchers / ...amid backlash) far more than the head.
    """
    return " ".join(title_tokens(title)[:5])


def cluster_key(url: str, title: str) -> str:
    """The identity of a STORY, across sources.

    URL first because it is exact. The prefix keeps the two namespaces apart,
    so a title that happens to look like a URL can never collide with one.
    """
    canon = canonical_url(url)
    if canon:
        return "u:" + canon
    key = title_key(title)
    return "t:" + key if key else ""


def same_story(a: dict, b: dict) -> bool:
    """Whether two rows are one story, for the fuzzy pass over title-only rows.

    Containment rather than equality: "OpenAI releases GPT-5" and "OpenAI
    releases GPT-5 to all paid tiers" are the same story, while sharing a
    mere prefix ("Ask HN: how do you ...") is not enough on its own - which
    is why at least three content words have to overlap.
    """
    ta, tb = set(title_tokens(a.get("title") or "")), set(title_tokens(b.get("title") or ""))
    if len(ta) < 3 or len(tb) < 3:
        return False
    return ta <= tb or tb <= ta
