from tech_news_curator.categorize import CATEGORIES, CATEGORY_ORDER, categorize, label

HINTS = {"robot_report": "robotics", "jvns": "devtools", "crunchbase": "startups"}


def cat(title, source="hackernews", **kw):
    return categorize({"title": title, "source": source, **kw}, HINTS)


def test_every_ordered_category_exists_and_other_is_not_orderable():
    assert set(CATEGORY_ORDER) <= set(CATEGORIES)
    assert "other" not in CATEGORY_ORDER, "'other' is the fallback, never a match"


def test_entity_names_carry_a_headline_that_never_says_the_topic():
    assert cat("Boston Dynamics teaches Atlas to fold laundry", "theverge") == "robotics"
    assert cat("Unitree's new humanoid does a backflip") == "robotics"


def test_theory_topics_land_in_research():
    assert cat("Model checking with TLA+ at scale") == "research"
    assert cat("A faster convex optimization solver") == "research"


def test_specific_beats_general_when_several_patterns_fire():
    # "reinforcement learning" is research; it also matches ai_ml's "learning".
    assert cat("Reinforcement learning for Lyapunov control") == "research"


def test_source_hints_apply_when_nothing_in_the_text_matches():
    assert cat("Some more things about Django I've been enjoying", "jvns") == "devtools"
    assert cat("An untagged post with no keywords", "robot_report") == "robotics"


def test_the_priority_order_settles_a_hint_against_a_match():
    # robot_report's hint is robotics, which outranks the devtools its
    # summary happens to match.
    assert cat("Our new SDK and API for arm control", "robot_report") == "robotics"
    # The other way round: crunchbase hints startups, but a robotics story
    # there is filed under the higher-priority bucket.
    assert cat("Humanoid robot maker raises a round", "crunchbase") == "robotics"


def test_tags_count_as_evidence():
    assert categorize({"title": "Workspace Models", "source": "arxiv",
                       "tags": ["cs.RO"]}, HINTS) == "research"


def test_unmatched_items_are_not_silently_dropped():
    assert cat("A story about nothing in particular") == "other"
    assert label("other") == "Everything Else"
    assert label("not_a_category") == "Everything Else"
