"""Turning ten incomparable crowd counts into one 0-100 number.

420 Hacker News points, 1,900 Reddit upvotes, 8,000 GitHub stars and 300
Bluesky likes are not on the same scale, do not have the same distribution,
and do not mean the same thing. Anything that compares them directly - a
fixed multiplier per source, a global sort on raw counts - is measuring the
size of a source's audience, not how much attention an item got.

So every score here is relative to the item's OWN source, over the same
window: half of it is where the item falls in that source's distribution,
half is how far above that source's typical item it reached. Both halves are
computed from the data in hand, so nothing needs recalibrating when a source
grows, goes quiet, or is added.

Sources that publish no counts get no number at all rather than an invented
one - `popularity` stays None and the UI sorts them last under "popular".
What those items get instead is `propagate`: an article that a feed carried
and Hacker News also ran inherits the Hacker News attention, because it is
the same story and the attention is real.
"""
from __future__ import annotations
import math

# How much a source's numbers are trusted as evidence of broad interest,
# as a multiplier on the final 0-100. This is the one deliberately editorial
# constant in the module: it says a story the Lobsters crowd pushed up is
# stronger evidence per-vote than a big Bluesky like count, because the
# voting populations differ in size and in what a vote costs.
SOURCE_TRUST = {
    "hackernews": 1.0,
    "reddit": 0.95,
    "lobsters": 0.9,
    "github": 0.85,   # stars accumulate; they lag rather than spike
    "bluesky": 0.7,   # a like is cheap, and engagement skews to opinion
    "twitter": 0.7,
}


def percentile_ranks(values: list) -> list:
    """Fractional rank in [0,1] of each value within its own list.

    Ties share the midpoint of the span they occupy, so a source where
    everything scored 50 gives every item 0.5 rather than an arbitrary order.
    """
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        midpoint = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = midpoint / (n - 1)
        i = j + 1
    return ranks


def _signal(row: dict):
    """The raw crowd count for a row, or None when the source publishes none.

    Comments are folded in at a discount: a thread with 600 replies and 400
    points is a bigger event than its point count alone suggests, but
    argument is not the same thing as approval, so it can't count equally.
    """
    metric = (row.get("metric") or "").strip()
    if metric == "rank":
        # Order-only source: position is the signal, inverted so that #1 is
        # the largest number, like every other signal here.
        rank = int(row.get("rank") or 0)
        return (1000.0 / rank) if rank > 0 else None
    if not metric:
        return None
    points = int(row.get("points") or 0)
    comments = int(row.get("comments") or 0)
    if points <= 0 and comments <= 0:
        return None
    return points + 0.5 * comments


def popularity_scores(rows: list) -> dict:
    """{row id: (0-100, one-line explanation)} for every row that has a signal."""
    by_source = {}
    for row in rows:
        value = _signal(row)
        if value is not None:
            by_source.setdefault(row["source"], []).append((row, value))

    out = {}
    for source, pairs in by_source.items():
        values = [v for _, v in pairs]
        ranks = percentile_ranks(values)
        # The reference point is the 90th percentile, not the maximum: one
        # freak 5,000-point story would otherwise compress an entire normal
        # day into the bottom of the scale.
        high = _quantile(values, 0.9) or max(values)
        trust = SOURCE_TRUST.get(source, 0.8)
        for (row, value), pct in zip(pairs, ranks):
            # Log ratio, because attention is multiplicative: the step from
            # 10 to 100 points matters far more than 1,000 to 1,090.
            magnitude = math.log1p(max(value, 0.0)) / math.log1p(max(high, 1.0))
            raw = 100.0 * (0.5 * pct + 0.5 * min(magnitude, 1.15))
            out[row["id"]] = (
                round(max(0.0, min(100.0, raw * trust)), 1),
                _reason(row, value, pct),
            )
    return out


def _reason(row: dict, value: float, pct: float) -> str:
    metric = (row.get("metric") or "").strip()
    if metric == "rank":
        return f"#{int(row.get('rank') or 0)} on {row['source']}"
    points, comments = int(row.get("points") or 0), int(row.get("comments") or 0)
    unit = metric or "points"
    bits = [f"{points} {unit}"] if points else []
    if comments:
        bits.append(f"{comments} comments")
    # "top 100%" is not a compliment. The standing is only worth stating
    # when it is actually a standing; below the median the raw count says
    # everything there is to say.
    if pct >= 0.5:
        bits.append(f"top {max(1, round((1 - pct) * 100))}% on {row['source']}")
    else:
        bits.append(f"on {row['source']}")
    return ", ".join(bits)


def _quantile(values: list, q: float):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def propagate(rows: list, scores: dict) -> dict:
    """Share each story's best popularity across every copy of it.

    A TechCrunch piece that also went to #2 on Hacker News did get that
    attention - the feed just can't see it. Without this, the aggregator's
    copy ranks and the publisher's copy (often the better-written one, with
    the image) sits unranked at the bottom.

    Returns a new dict; the input is not modified.
    """
    best = {}
    for row in rows:
        key = row.get("cluster_key")
        if not key:
            continue
        score = scores.get(row["id"])
        if score and (key not in best or score[0] > best[key][0]):
            best[key] = (score[0], row["source"])

    out = dict(scores)
    for row in rows:
        key = row.get("cluster_key")
        if not key or key not in best:
            continue
        value, from_source = best[key]
        current = out.get(row["id"])
        if current and current[0] >= value:
            continue
        # Say where an inherited number came from: a reader looking at a
        # Verge item marked 88 deserves to know that is Hacker News's 88.
        note = f"via {from_source}" if from_source != row["source"] else ""
        if current:
            out[row["id"]] = (value, f"{current[1]} ({note})" if note else current[1])
        elif note:
            out[row["id"]] = (value, f"same story {note}")
    return out
