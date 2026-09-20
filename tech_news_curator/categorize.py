"""Putting each item in one bucket, twice over.

The taxonomy is fixed and small on purpose. Free-form tags from ten sources
don't line up ("ML" / "machine-learning" / "AI"), so they can't drive a
filter UI; a closed list can, and it is also what the LLM is constrained to
return, which keeps the model's categories and the heuristic's identical.

Two passes, same split as the ranker: keywords run always and free, the LLM
overwrites them when it runs. The keyword pass is not a fallback nobody
sees - it is what categorizes the long tail the LLM never gets sent.
"""
from __future__ import annotations
import re

# id -> (label, ordered keyword patterns). Order matters: the first match in
# CATEGORY_ORDER wins, so specific buckets are checked before general ones.
CATEGORIES = {
    "robotics": "Robotics & Autonomy",
    "ai_ml": "AI & ML",
    "research": "Research & Theory",
    "startups": "Startups & Funding",
    "finance": "Finance & Markets",
    "open_source": "Open Source",
    "devtools": "Dev Tools & Productivity",
    "hardware": "Hardware & Chips",
    "security": "Security",
    "policy": "Policy & Society",
    "rumors": "Rumors & Leaks",
    "other": "Everything Else",
}

# Priority order, highest first - not an alphabet and not a specificity
# metric, just the order in which an item that matches several buckets
# should be filed. Robotics and research sit above AI because a
# control-theory paper about a quadruped matches "learning" too, and the
# narrower bucket is the more useful place to find it. "other" is absent:
# it is the fallback, never a match.
CATEGORY_ORDER = (
    "robotics", "research", "startups", "finance", "rumors", "security",
    "hardware", "open_source", "devtools", "ai_ml", "policy",
)

PATTERNS = {
    # Entity names carry as much weight as the topic words: a headline about
    # Atlas folding laundry never says "robot", and that is precisely the
    # story this bucket exists for.
    "robotics": r"\b(robot|robotics|humanoid|drone|uav|quadruped|manipulat|teleoperat|"
                r"autonomous vehicle|self-driving|waymo|zoox|cruise|tesla fsd|slam|"
                r"lidar|actuator|end-effector|embodied|boston dynamics|figure ai|"
                r"unitree|agility robotics|anduril|skydio|nuro|kuka|fanuc|cobot)\b",
    "research": r"\b(arxiv|preprint|paper|theorem|proof|formal method|model checking|"
                r"verification|tla\+|coq|lean 4|isabelle|smt solver|optimi[sz]ation|"
                r"convex|gradient descent|control theory|mpc|kalman|lyapunov|"
                r"reinforcement learning|benchmark result|ablation)\b",
    "startups": r"\b(series [a-e]\b|seed round|pre-seed|raises?\s+\$|funding round|"
                r"valuation|y combinator|yc [swf]\d|startup|acqui-?hire|term sheet|"
                r"venture|vc firm|founded by)\b",
    "finance": r"\b(ipo|earnings|revenue|market cap|stock|shares?|nasdaq|antitrust|"
                r"acquisition|acquires?|merger|billion|q[1-4] results|guidance|"
                r"short seller|buyback)\b",
    "rumors": r"\b(rumor|rumour|leak(ed|s)?|reportedly|sources say|allegedly|"
              r"is said to|expected to announce|renders? show|next-gen \w+ may)\b",
    "security": r"\b(vulnerabilit|cve-\d|exploit|zero-day|0-day|breach|ransomware|"
                r"malware|phishing|backdoor|patch tuesday|rce\b|supply chain attack)\b",
    "hardware": r"\b(chip|silicon|gpu|tpu|npu|semiconductor|fab\b|tsmc|nvidia|amd|"
                r"arm\b|risc-v|wafer|nanometer|\bnm process|soc\b|memory bandwidth|hbm)\b",
    "open_source": r"\b(open[- ]source|foss|oss\b|apache 2|mit license|gpl|agpl|"
                   r"fork(ed)?|maintainer|contributor|upstream|release \d+\.\d|"
                   r"v\d+\.\d+\.\d+|github\.com)\b",
    "devtools": r"\b(ide\b|editor|cli\b|framework|library|sdk\b|api\b|compiler|"
                r"linter|debugger|build system|ci/cd|devops|kubernetes|docker|"
                r"productivity|workflow|terminal|shell|plugin|extension)\b",
    "ai_ml": r"\b(ai\b|a\.i\.|llm|gpt|claude|gemini|llama|mistral|transformer|"
             r"neural|machine learning|deep learning|fine-?tun|inference|embedding|"
             r"diffusion|agentic|rag\b|prompt|token|model weights|openai|anthropic)\b",
    "policy": r"\b(regulat|legislat|congress|senate|eu ai act|gdpr|ftc|doj|lawsuit|"
              r"court|ban(ned|s)?\b|copyright|privacy|surveillance|export control)\b",
}

COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}

# Sources whose every item belongs to one bucket regardless of wording.
SOURCE_HINTS = {
    "arxiv": "research",
    "github": "open_source",
}


def categorize(item: dict, feed_hints: dict = None) -> str:
    """Best-effort category id for one item. Always returns something."""
    source = item.get("source") or ""
    hint = SOURCE_HINTS.get(source) or (feed_hints or {}).get(source, "")

    # Title, summary and the source's own tags all count as evidence; a
    # tag list like ["cs.RO", "Python"] is often the clearest signal there is.
    tags = item.get("tags") or []
    haystack = " ".join([
        item.get("title") or "",
        (item.get("summary") or "")[:300],
        " ".join(tags if isinstance(tags, list) else [str(tags)]),
    ])

    matched = next((c for c in CATEGORY_ORDER if COMPILED[c].search(haystack)), "")
    if not matched:
        return hint or "other"
    if not hint or hint not in CATEGORY_ORDER:
        return matched
    # Both fired, so the priority order decides. This is what keeps a plain
    # robot story on a robotics site filed as "robotics" when a stray "api"
    # in its summary also matched devtools.
    return min(matched, hint, key=CATEGORY_ORDER.index)


def label(cat: str) -> str:
    return CATEGORIES.get(cat, CATEGORIES["other"])
