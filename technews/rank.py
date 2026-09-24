"""Scoring, so a day's 900 items become a list worth reading top-down.

Two rankers, the same layering as the sibling sfevents project:

- `heuristic_scores` is deterministic, offline and free. It answers "how big
  a deal is this generally?" from signals that need no model: normalized
  popularity, how many independent sources ran it, how fresh it is, and how
  much the source is trusted. It runs every time and is the floor when
  there's no key, no network, or a spent quota.

- `llm_scores` answers the question that actually matters here - "would THIS
  person open this?" - by reading profile.md. It also assigns the category,
  in the same call, because the model has already read the item and a second
  request to classify it would double the cost for nothing.

Which model does the rating is a swappable provider - Gemini, Claude, Groq,
or a local Ollama model - see PROVIDERS. `llm_scores_chain` adds fallbacks:
whatever the first provider leaves unscored (a 503, a spent quota) goes to
the next.

Scores are cached by a hash of (profile, model), so a daily run only pays
for items it has never seen. Editing profile.md changes the hash and
re-scores everything once, which is the intended way to retune the feed.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .categorize import CATEGORIES

DEFAULT_PROFILE = Path(__file__).parent.parent / "profile.md"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-3.6-flash"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
# Small enough (~3GB at Q4) to run on a GitHub Actions runner's CPU. The env
# var lets the workflow pull and use the same model from one setting.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or "qwen3.5:4b"

# How much a source is trusted to be worth surfacing at all, independent of
# how many votes a given item got. Distinct from popularity.SOURCE_TRUST,
# which scales the vote counts; this is about editorial signal-to-noise.
SOURCE_WEIGHT = {
    "hackernews": 5,
    "lobsters": 5,
    "techmeme": 5,     # an aggregator of aggregators: appearing here is itself news
    "arxiv": 3,
    "github": 3,
    "reddit": 2,
    "bluesky": -3,     # highest noise floor of anything here
    "twitter": -3,
    "producthunt": -2, # launches are relentless and mostly not for this reader
}

# Items whose titles are structurally low-information, whatever the source.
FILLER_RE = re.compile(
    r"\b(sponsored|advertisement|deals?|discount|coupon|sale|giveaway|"
    r"newsletter|roundup|weekly digest|open thread|daily discussion|"
    r"who is hiring|what are you working on)\b",
    re.I,
)


# --------------------------------------------------------------------------
# heuristic ranker
# --------------------------------------------------------------------------

def hours_old(row: dict, now: datetime = None) -> float:
    """Age in hours, from the published date or else when we first saw it."""
    now = now or datetime.now(timezone.utc)
    stamp = row.get("published_ts") or row.get("first_seen")
    if not stamp:
        return 48.0
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return 48.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds() / 3600.0)


def freshness(hours: float) -> float:
    """1.0 for something posted now, halving every 36 hours.

    News decays in a way events do not - a two-day-old story is not two
    days' less relevant, it is stale. Exponential rather than linear decay
    is what keeps a thousand-point story from holding the top of the page
    all week.
    """
    return 0.5 ** (hours / 36.0)


def cluster_sources(rows: list) -> dict:
    """{row id: set of sources that carried this same story}."""
    members = {}
    for row in rows:
        key = row.get("cluster_key")
        if key:
            members.setdefault(key, set()).add(row["source"])
    return {
        row["id"]: members.get(row.get("cluster_key") or "", {row["source"]})
        for row in rows
    }


def heuristic_scores(rows: list, now: datetime = None) -> dict:
    """Score every row 0-100 on general notability. Pure function of the input."""
    agreement = cluster_sources(rows)
    out = {}
    for row in rows:
        why = []
        # Popularity, where it exists, is the backbone; without it the item
        # starts from a neutral prior rather than from zero, because "no
        # source counted votes" is not evidence of being uninteresting.
        pop = row.get("popularity")
        if pop is None:
            score = 30.0
        else:
            score = 0.7 * float(pop)
            if pop >= 70:
                why.append("widely upvoted")

        sources = agreement.get(row["id"], set())
        if len(sources) > 1:
            score += min(24, 12 * (len(sources) - 1))
            why.append(f"carried by {len(sources)} sources")

        age = hours_old(row, now)
        score *= 0.6 + 0.4 * freshness(age)
        if age <= 12:
            why.append("fresh")

        score += SOURCE_WEIGHT.get(row["source"], 0)
        if FILLER_RE.search(row.get("title") or ""):
            score -= 20
            why.append("routine/recurring post")
        if row.get("summary"):
            score += 2  # something for the reader and the ranker to go on

        out[row["id"]] = (
            round(max(0.0, min(100.0, score)), 1),
            ", ".join(why[:3]) or "baseline listing",
        )
    return out


# --------------------------------------------------------------------------
# LLM ranker
# --------------------------------------------------------------------------

def load_profile(path=DEFAULT_PROFILE) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"no profile at {p}. Copy profile.example.md to profile.md and edit it - "
            "the LLM ranker needs to know who it's ranking for."
        )
    return strip_comments(p.read_text())


def strip_comments(text: str) -> str:
    """Drop HTML comments so editing notes never reach the model.

    profile.md is a prompt, not documentation: every word in it is sent.
    Instructions written for the human reader would otherwise be read as
    facts about the person.
    """
    without = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", without).strip()


CATEGORY_LIST = ", ".join(f"{k} ({v})" for k, v in CATEGORIES.items())


def profile_hash(profile: str, model: str) -> str:
    """Identifies a (profile, model, taxonomy) triple, so an edit to any of
    them forces a re-rank.

    The taxonomy is in here because the cached row holds a category as well
    as a score. Leave it out and splitting a category would leave every item
    already scored sitting in a bucket that no longer means what it did.
    """
    key = f"{model}\x00{profile}\x00{CATEGORY_LIST}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _item_line(i: int, row: dict) -> str:
    bits = [f"{i}. {row['title']}"]
    bits.append(f"src={row['source']}")
    pop = row.get("popularity")
    if pop is not None:
        # The model is told how popular something is, but the prompt is
        # explicit that popularity is context, not the thing being scored.
        bits.append(f"pop={int(pop)}")
    if row.get("points"):
        bits.append(f"{row.get('metric') or 'points'}={row['points']}")
    host = ""
    try:
        host = urllib.parse.urlsplit(row.get("url") or "").netloc.replace("www.", "")
    except ValueError:
        pass
    if host:
        bits.append(f"site={host}")
    tags = row.get("tags")
    if tags:
        bits.append("tags=" + ",".join((tags if isinstance(tags, list) else [tags])[:5]))
    summary = re.sub(r"\s+", " ", (row.get("summary") or ""))[:200]
    if summary:
        bits.append(f"note={summary}")
    return " | ".join(bits)


PROMPT = """You are triaging a day of tech news for one specific person, whose \
profile is below. This is their daily reading list, so score each item on \
whether THEY would open it - not on how important it is in general.

PERSON'S PROFILE
{profile}

SCORING
90-100 = they would stop and read this today
70-89  = clearly in their interests, worth surfacing
40-69  = plausible, depends on the day
10-39  = weak match
0-9    = noise for this person

A famous story outside their interests scores low. A small post squarely \
inside them scores high. The `pop=` figure is how much attention the item \
got elsewhere: use it to break ties, never as the score itself - a viral \
story they don't care about is still noise. Penalise low-information items \
(listicles, recurring threads, press releases, launch spam). If the profile \
is silent on something, score it in the middle rather than guessing.

CATEGORY - pick exactly one id from: {categories}
Prefer the most specific id that fits. The robot_* ids are for work about \
robots and autonomous machines: an optimization or multi-agent or \
foundation-model item that is not about robots belongs in research or ai_ml, \
not in a robot_* bucket.

Return ONLY a JSON array, no prose, no code fence:
[{{"i": <item number>, "score": <integer 0-100>, "category": "<category id>", \
"reason": "<max 12 words, specific to this person>"}}]

ITEMS
{items}"""


class ProviderError(ValueError):
    """An API call failed and the provider said why.

    Subclasses ValueError so llm_scores' per-batch handling catches it: one
    bad batch is reported and skipped, not fatal.
    """

    def __init__(self, message, status=None, retry_after=None, daily=False):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        # A per-day quota won't clear by waiting a minute, so retrying only
        # burns tomorrow's allowance.
        self.daily = daily


def _redact(text: str) -> str:
    """Never let an API key reach a log. Gemini puts its key in the URL."""
    return re.sub(r"(key=)[^&\s\"']+", r"\1***", text)


USER_AGENT = "technews/0.1 (+https://github.com/watakandai/technews)"


def _post_json(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        # Groq sits behind Cloudflare, which rejects urllib's default
        # "Python-urllib/3.x" agent with a bare 403 (error code 1010).
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # The status alone ("400 Bad Request") doesn't say whether the key,
        # the model or the payload is wrong - the body does.
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - body already consumed
            body = ""
        detail = _redact(body)[:400] or exc.reason
        quota = re.search(r'"quotaId"\s*:\s*"([^"]+)"', body)
        if quota:
            detail = f"quota {quota.group(1)} exhausted"
        # Groq names the limit in prose: "... on tokens per day (TPD)".
        daily = (bool(quota) and "PerDay" in quota.group(1)) or (
            exc.code == 429 and re.search(r"per day", body, re.I) is not None
        )
        raise ProviderError(
            f"HTTP {exc.code}: {detail}",
            status=exc.code,
            retry_after=_retry_after(exc.headers, body),
            daily=daily,
        ) from None


def _retry_after(headers, body: str):
    """How long the provider asked us to wait, if it said.

    Anthropic sends a retry-after header; Gemini puts "retryDelay": "37s"
    in the error body instead.
    """
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body or "")
    return float(match.group(1)) if match else None


def _call_gemini(prompt: str, model: str, api_key: str, timeout: int) -> str:
    # Gemini takes the key as a query parameter rather than a header.
    data = _post_json(
        GEMINI_URL.format(model=model) + f"?key={urllib.parse.quote(api_key)}",
        {},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            # No temperature override: Google warns that going below the
            # default on Gemini 3 models can cause looping. Thinking stays
            # low - triage against a profile isn't a reasoning problem, and
            # thinking tokens are billed as output.
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
        },
        timeout,
    )
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"gemini returned no candidates: {str(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _call_anthropic(prompt: str, model: str, api_key: str, timeout: int) -> str:
    data = _post_json(
        ANTHROPIC_URL,
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        {"model": model, "max_tokens": 8192,
         "messages": [{"role": "user", "content": prompt}]},
        timeout,
    )
    return "".join(
        block.get("text", "") for block in data.get("content", [])
        if block.get("type") == "text"
    )


def _chat_completions(url: str, prompt: str, model: str, api_key: str,
                      timeout: int, **extra) -> str:
    """The OpenAI-style request most other providers (Groq included) accept."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _post_json(
        url,
        headers,
        {"model": model, "messages": [{"role": "user", "content": prompt}], **extra},
        timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in reply: {str(data)[:200]}")
    return choices[0].get("message", {}).get("content") or ""


def _call_groq(prompt: str, model: str, api_key: str, timeout: int) -> str:
    # gpt-oss is a reasoning model. Low effort keeps the hidden reasoning -
    # which counts against Groq's tokens-per-minute cap - short, and
    # include_reasoning=False keeps it out of the reply we parse.
    return _chat_completions(
        GROQ_URL, prompt, model, api_key, timeout,
        reasoning_effort="low", include_reasoning=False,
        max_completion_tokens=4096,
    )


def _call_ollama(prompt: str, model: str, host: str, timeout: int) -> str:
    """A model on a local (or self-hosted) Ollama server - no key, no quota.

    `host` comes from OLLAMA_HOST, which doubles as the "key": set means a
    server is up. Ollama's own CLI accepts it without a scheme, so this does
    too. The native /api/chat endpoint is used rather than Ollama's
    OpenAI-style one because only it can raise the context window, and
    Ollama's small default would silently cut a batch off mid-list.
    """
    base = host.strip().rstrip("/")
    if "://" not in base:
        base = f"http://{base}"
    data = _post_json(
        f"{base}/api/chat",
        {},
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Qwen 3.5 thinks by default; on a CPU that costs minutes a batch
            # and triage doesn't need it.
            "think": False,
            "options": {"num_ctx": 8192},
        },
        timeout,
    )
    return data.get("message", {}).get("content") or ""


# (env var holding the key, default model, call function). The signature is
# (prompt, model, key, timeout) -> reply text, so adding a provider is one
# function plus one line - nothing else in this module, the CLI or the
# workflow needs to know which one is in use.
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", GEMINI_MODEL, _call_gemini),
    "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_MODEL, _call_anthropic),
    "groq": ("GROQ_API_KEY", GROQ_MODEL, _call_groq),
    "ollama": ("OLLAMA_HOST", OLLAMA_MODEL, _call_ollama),
}

# (items per request, seconds between requests) for a provider when it runs
# as a fallback, where the CLI's --batch-size/--min-interval (tuned for the
# primary) don't apply. Groq's free tier caps tokens per minute (8K on
# gpt-oss-120b), not requests per day: a 15-item request with its category
# list is ~3K tokens, so one every 30 seconds stays under the cap.
FALLBACK_PACING = {
    "groq": (15, 30.0),
    # No rate limit to respect, but a 4B model on a CPU slows down as the
    # prompt grows - small batches keep each request to a minute or so.
    "ollama": (10, 0.0),
}

# Per-request timeouts for providers slower than the default. A CPU-only
# Ollama can take minutes on one batch.
PROVIDER_TIMEOUT = {
    "ollama": 600,
}


def parse_results(text: str, batch_size: int) -> dict:
    """Pull the JSON array out of a model reply, tolerating fences and prose."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"no JSON array in model reply: {text[:200]!r}")
    parsed = json.loads(match.group(0))
    out = {}
    for entry in parsed:
        try:
            idx = int(entry["i"])
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= idx <= batch_size:
            continue
        category = str(entry.get("category") or "").strip().lower()
        out[idx] = {
            "score": max(0.0, min(100.0, score)),
            # An invented category would quietly break the filter UI, so
            # anything off-list is dropped and the keyword pass stands.
            "category": category if category in CATEGORIES else "",
            "reason": re.sub(r"\s+", " ", str(entry.get("reason", ""))).strip()[:120],
        }
    return out


# Waits between retries of a rate-limited batch, when the provider doesn't
# say how long. Free-tier limits are per minute, so the last wait covers a
# full window.
RETRY_WAITS = (15, 30, 65)


def _call_with_retry(call, prompt, model, key, timeout, sleep):
    for wait in RETRY_WAITS + (None,):
        try:
            return call(prompt, model, key, timeout)
        except ProviderError as exc:
            if exc.status not in (429, 500, 503) or exc.daily or wait is None:
                raise
            sleep(min(exc.retry_after or wait, 120))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            # A reset connection or a timeout is as transient as a 503.
            if wait is None:
                raise
            sleep(wait)


def llm_scores(rows, profile, *, provider="gemini", model=None, batch_size=40,
               timeout=120, min_interval=0, on_progress=None,
               sleep=time.sleep, clock=time.monotonic) -> dict:
    """Score and categorize rows against the profile. {row id: result dict}.

    Batches are independent: one failing batch is reported and skipped
    rather than losing the whole run, so a rate limit halfway through still
    leaves the batches that succeeded. A 429 is waited out and retried
    first. min_interval spaces batches out, so a run stays under a
    per-minute limit instead of hitting it and waiting.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    env_var, default_model, call = PROVIDERS[provider]
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(f"{env_var} is not set, so provider {provider!r} can't be used")
    model = model or default_model

    out = {}
    rate_limited = False
    last_call = None
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if rate_limited:
            # Once retries couldn't clear a 429, this is a daily quota, not
            # a per-minute one - every later batch would fail the same way.
            # They stay unscored and tomorrow's run picks them up.
            if on_progress:
                on_progress(start, len(batch), "skipped (rate limited)")
            continue
        prompt = PROMPT.format(
            profile=profile,
            categories=CATEGORY_LIST,
            items="\n".join(_item_line(i, r) for i, r in enumerate(batch, 1)),
        )
        if min_interval and last_call is not None:
            wait = min_interval - (clock() - last_call)
            if wait > 0:
                sleep(wait)
        last_call = clock()
        try:
            reply = _call_with_retry(call, prompt, model, key, timeout, sleep)
            scored = parse_results(reply, len(batch))
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError) as exc:
            if getattr(exc, "status", None) == 429:
                rate_limited = True
            if on_progress:
                on_progress(start, len(batch), f"FAILED ({type(exc).__name__}: {exc})")
            continue
        for idx, result in scored.items():
            out[batch[idx - 1]["id"]] = result
        if on_progress:
            on_progress(start, len(batch), f"{len(scored)} scored")
    return out


def llm_scores_chain(rows, profile, providers, *, model=None, batch_size=40,
                     min_interval=0, on_progress=None, on_provider=None,
                     **kwargs) -> dict:
    """Score rows with providers[0], then hand what it missed to the next.

    Returns {"provider:model": {row id: result dict}}, so each score keeps a
    record of which model actually gave it. `model`, `batch_size` and
    `min_interval` apply to the first provider; fallbacks use their own
    default model and FALLBACK_PACING. A fallback with no key configured is
    skipped (reported through on_provider) rather than failing the run -
    only an unusable first provider is an error.
    """
    for name in providers:
        if name not in PROVIDERS:
            raise ValueError(f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")

    results = {}
    pending = rows
    for n, name in enumerate(providers):
        if not pending:
            break
        env_var, default_model, _ = PROVIDERS[name]
        if n and not os.environ.get(env_var, "").strip():
            if on_provider:
                on_provider(name, None, len(pending), f"skipped ({env_var} not set)")
            continue
        use_model = model if n == 0 and model else default_model
        size, interval = (
            (batch_size, min_interval) if n == 0
            else FALLBACK_PACING.get(name, (batch_size, min_interval))
        )
        if on_provider:
            note = "" if n == 0 else f"falling back for {len(pending)} unscored items"
            on_provider(name, use_model, len(pending), note)
        if name in PROVIDER_TIMEOUT:
            kwargs = {**kwargs, "timeout": PROVIDER_TIMEOUT[name]}
        got = llm_scores(
            pending, profile, provider=name, model=use_model,
            batch_size=size, min_interval=interval, on_progress=on_progress, **kwargs,
        )
        if got:
            results[f"{name}:{use_model}"] = got
        pending = [r for r in pending if r["id"] not in got]
    return results
