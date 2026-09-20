"""Putting each item in one bucket, twice over.

The taxonomy is fixed and small on purpose. Free-form tags from ten sources
don't line up ("ML" / "machine-learning" / "AI"), so they can't drive a
filter UI; a closed list can, and it is also what the LLM is constrained to
return, which keeps the model's categories and the heuristic's identical.

Two passes, same split as the ranker: keywords run always and free, the LLM
overwrites them when it runs. The keyword pass is not a fallback nobody
sees - it is what categorizes the long tail the LLM never gets sent.

Research is split by subfield rather than kept as one bucket, because "a
paper" is not a useful thing to be told daily: planning and control,
optimization, formal methods, multi-agent planning and robot learning are
different reading, and the point of the page is to see which of them had a
day. Those five are scoped to robotics - see GATED - and `research` keeps
everything else academic.
"""
from __future__ import annotations
import re

CATEGORIES = {
    "robotics": "Robotics & Autonomy",
    "robot_planning": "Robot Planning & Control",
    "robot_optimization": "Robot Optimization",
    "robot_formal": "Robot Formal Methods",
    "robot_multiagent": "Robot Multi-Agent Planning",
    "robot_ai": "Robot Learning & Foundation Models",
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
# should be filed.
#
# The five subfields outrank `robotics` itself: almost every robotics paper
# also says "robot", so filing on that word first would empty the subfield
# buckets and leave one undifferentiated pile. `robot_ai` comes last of the
# five because "learning" appears in control papers far more often than
# "Lyapunov" appears in learning papers. `policy` outranks `ai_ml` for the
# same reason in reverse: "ai" is the single most promiscuous token here, and
# an AI Act story is about the Act. `other` is absent: it is the fallback,
# never a match.
CATEGORY_ORDER = (
    "robot_formal", "robot_multiagent", "robot_optimization",
    "robot_planning", "robot_ai",
    "robotics", "research", "startups", "finance", "rumors", "security",
    "hardware", "open_source", "devtools", "policy", "ai_ml",
)


def _re(body: str) -> re.Pattern:
    """Compile one bucket's alternation, anchored at both ends.

    The trailing `(?!\\w)` is doing the job you would expect `\\b` to do, and
    the reason for the difference is worth the line: `\\b(robot|...)\\b` does
    not match "robots" (the boundary after "robot" fails on the "s"), and
    does not match "tla+" either (the boundary after "+" needs a word
    character before it). A stem written as `robot\\w*` plus `(?!\\w)` matches
    the inflections, and punctuation-final terms survive.
    """
    return re.compile(rf"\b(?:{body})(?!\w)", re.I)


PATTERNS = {
    # Entity names carry as much weight as the topic words: a headline about
    # Atlas folding laundry never says "robot", and that is precisely the
    # story this bucket exists for. arXiv's own subject codes are in here
    # too - on a preprint they are the clearest signal there is.
    "robotics": r"robot\w*|humanoid\w*|quadruped\w*|legged|locomotion|gait|"
                r"exoskeleton\w*|(?<!market )(?<!price )manipulat\w*|grasp\w*|"
                r"drone\w*|uav|teleoperat\w*|end-effector\w*|embodied|"
                r"autonomous vehicle\w*|self-driving|waymo|zoox|cruise|tesla fsd|"
                r"slam|lidar|actuator\w*|cobot\w*|cs\.ro|"
                r"boston dynamics|figure ai|unitree|agility robotics|anduril|"
                r"skydio|nuro|kuka|fanuc",

    # --- the five robotics research subfields --------------------------
    # What each of these matches on its own is deliberately narrow: terms
    # that mean robotics wherever they appear. The overloaded half of each
    # vocabulary lives in GATED below.
    "robot_planning": r"motion planning|path planning|task and motion planning|tamp|"
                      r"whole-?body control|footstep\w*|visual servoing|"
                      r"impedance control|admittance control|inverse kinematics|"
                      r"forward kinematics|collision avoidance|obstacle avoidance|"
                      r"navigation stack|motion primitive\w*|behavio(?:u)?r tree\w*|"
                      r"rrt\*?|prm|sampling-based planning|trajectory tracking",

    "robot_optimization": r"trajectory optimi[sz]\w*|direct collocation|collocation|"
                          r"contact-implicit|differentiable simulation|"
                          r"bundle adjustment|factor graph\w*|"
                          r"ipopt|osqp|casadi|acados|crocoddyl|drake",

    "robot_formal": r"control barrier function\w*|cbf|barrier certificate\w*|"
                    r"reachability analysis|reachable set\w*|hybrid automat\w*|"
                    r"signal temporal logic|linear temporal logic|"
                    r"reactive synthesis|correct-by-construction|"
                    r"runtime verification|safety verification|"
                    r"provably safe|certified safe|safety shield\w*",

    "robot_multiagent": r"multi-?robot\w*|swarm\w*|mapf|multi-?agent path ?finding|"
                        r"formation control|cooperative manipulat\w*|"
                        r"fleet coordination|multi-?agent planning|"
                        r"decentrali[sz]ed planning|task allocation",

    "robot_ai": r"vision-language-action|vla|openvla|rt-[12x]|"
                r"diffusion policy|behavio(?:u)?r cloning|imitation learning|"
                r"sim-?to-?real|sim2real|visuomotor|robot learning|embodied ai|"
                r"manipulation polic\w*|robot foundation model\w*|"
                r"learning from demonstration",
    # -------------------------------------------------------------------

    # Everything academic that is not one of the five above. Terms that also
    # appear in GATED are intentional duplicates: with robotics context the
    # item files as a subfield, without it, here.
    "research": r"arxiv|preprint\w*|paper\w*|theorem\w*|proof\w*|formal method\w*|"
                r"model checking|verification|tla\+|coq|lean 4|isabelle|"
                r"smt solver\w*|optimi[sz]ation\w*|convex|gradient descent|"
                r"control theory|mpc|kalman|lyapunov|reinforcement learning|"
                r"benchmark result\w*|ablation\w*",

    "startups": r"series [a-e]|seed round|pre-seed|raises?\s+\$|funding round|"
                r"valuation\w*|y combinator|yc [swf]\d|startup\w*|acqui-?hire\w*|"
                r"term sheet|venture|vc firm|founded by",

    "finance": r"ipo|earnings|revenue|market cap|stock\w*|shares?|nasdaq|"
               r"antitrust|acquisition\w*|acquires?|merger\w*|billion|"
               r"q[1-4] results|guidance|short seller\w*|buyback\w*",

    "rumors": r"rumou?r\w*|leak(?:ed|s|ing)?|reportedly|sources say|allegedly|"
              r"is said to|expected to announce|renders? show|next-gen \w+ may",

    "security": r"vulnerabilit\w*|cve-\d+|exploit\w*|zero-day|0-day|breach\w*|"
                r"ransomware|malware|phishing|backdoor\w*|patch tuesday|rce|"
                r"supply chain attack\w*",

    "hardware": r"chip\w*|silicon|gpu\w*|tpu\w*|npu\w*|semiconductor\w*|fab|tsmc|"
                r"nvidia|amd|arm|risc-v|wafer\w*|nanometer\w*|\d+\s*nm process|"
                r"soc|memory bandwidth|hbm\d*",

    "open_source": r"open[- ]source\w*|foss|oss|apache 2|mit license|gpl|agpl|"
                   r"fork(?:ed|s|ing)?|maintainer\w*|contributor\w*|upstream|"
                   r"release \d+\.\d|v\d+\.\d+\.\d+|github\.com",

    "devtools": r"ide|editor\w*|cli|framework\w*|librar(?:y|ies)|sdk|api\w*|"
                r"compiler\w*|linter\w*|debugger\w*|build system\w*|ci/cd|devops|"
                r"kubernetes|docker|productivity|workflow\w*|terminal\w*|shell|"
                r"plugin\w*|extension\w*",

    "ai_ml": r"ai|a\.i\.|llms?|gpt|claude|gemini|llama|mistral|transformer\w*|"
             r"foundation model\w*|vlm\w*|multimodal|"
             r"neural|machine learning|deep learning|fine-?tun\w*|inference|"
             r"embedding\w*|diffusion|agentic|rag|prompt\w*|token\w*|"
             r"model weights|openai|anthropic",

    "policy": r"regulat\w*|legislat\w*|congress|senate|eu ai act|gdpr|ftc|doj|"
              r"lawsuit\w*|court|ban(?:ned|s)?|copyright|privacy|surveillance|"
              r"export control\w*",
}

# The overloaded half of each subfield's vocabulary. "optimization",
# "multi-agent", "verification" and "foundation model" are four of the most
# reused phrases in tech, and a bucket that claimed every one of them would
# swallow the compiler, the distributed-systems and the LLM-agent story
# alike. These fire only when the same item also reads as robotics, so an
# MPC paper about a quadruped files as Robot Planning & Control and one
# about a chemical plant falls through to Research & Theory.
GATED = {
    "robot_planning": r"planner\w*|planning|optimal control|feedback control|"
                      r"control theory|controller\w*|trajector(?:y|ies)|"
                      r"navigation|state estimation|kalman|lyapunov|"
                      r"model predictive control|mpc|lqr|ilqr|ddp|"
                      r"stability analysis|eess\.sy|cs\.sy",

    "robot_optimization": r"optimi[sz]ation\w*|optimi[sz]er\w*|optimi[sz]ing|convex|"
                          r"nonlinear program\w*|mixed-?integer|milp|qp|sdp|"
                          r"semidefinite|sum-of-squares|gradient descent|"
                          r"interior point|cost function\w*|solver\w*|math\.oc",

    "robot_formal": r"formal method\w*|formal verification|model checking|"
                    r"verification|temporal logic|ltl|stl|tla\+|coq|lean 4|"
                    r"isabelle|smt solver\w*|theorem prover\w*|"
                    r"correctness guarantee\w*|invariant\w*|cs\.lo|cs\.fl",

    "robot_multiagent": r"multi-?agent\w*|cs\.ma|dec-pomdp|game theor\w*|consensus|"
                        r"distributed control|coordination|auction-based",

    "robot_ai": r"foundation model\w*|vlm\w*|large language model\w*|llms?|"
                r"transformer\w*|reinforcement learning|polic(?:y|ies)|"
                r"end-to-end|self-supervised|neural network\w*|world model\w*|"
                r"diffusion model\w*|pretrain\w*|cs\.ai|cs\.lg",
}

COMPILED = {k: _re(v) for k, v in PATTERNS.items()}
COMPILED_GATED = {k: _re(v) for k, v in GATED.items()}

# What "reads as robotics" means for GATED. It is the robotics bucket's own
# vocabulary, so there is exactly one definition of the subject in this file.
ROBOTICS_CONTEXT = COMPILED["robotics"]

# Sources whose every item belongs to one bucket regardless of wording.
SOURCE_HINTS = {
    "arxiv": "research",
    "github": "open_source",
}


def _fires(cat: str, haystack: str, robotic: bool) -> bool:
    """Does this bucket claim the item? Gated terms need robotics context."""
    if COMPILED[cat].search(haystack):
        return True
    gated = COMPILED_GATED.get(cat)
    return bool(robotic and gated is not None and gated.search(haystack))


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

    robotic = bool(ROBOTICS_CONTEXT.search(haystack))
    matched = next((c for c in CATEGORY_ORDER if _fires(c, haystack, robotic)), "")
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
