from technews.categorize import CATEGORIES, CATEGORY_ORDER, categorize, label

HINTS = {"robot_report": "robotics", "jvns": "devtools", "crunchbase": "startups"}


def cat(title, source="hackernews", **kw):
    return categorize({"title": title, "source": source, **kw}, HINTS)


def test_every_ordered_category_exists_and_other_is_not_orderable():
    assert set(CATEGORY_ORDER) <= set(CATEGORIES)
    assert "other" not in CATEGORY_ORDER, "'other' is the fallback, never a match"


def test_entity_names_carry_a_headline_that_never_says_the_topic():
    assert cat("Boston Dynamics teaches Atlas to fold laundry", "theverge") == "robotics"
    assert cat("Unitree's new humanoid does a backflip") == "robotics"


def test_theory_topics_land_in_research_when_no_robot_is_involved():
    assert cat("Model checking with TLA+ at scale") == "research"
    assert cat("A faster convex optimization solver") == "research"
    assert cat("MPC for chemical process control") == "research"


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
    # arXiv's hint is "research", but its own subject code is the stronger
    # statement: a cs.RO paper is robotics whatever its title says.
    assert categorize({"title": "Workspace Models", "source": "arxiv",
                       "tags": ["cs.RO"]}, HINTS) == "robotics"
    assert categorize({"title": "Untitled", "source": "arxiv",
                       "tags": ["cs.RO", "math.OC"]}, HINTS) == "robot_optimization"


def test_unmatched_items_are_not_silently_dropped():
    assert cat("A story about nothing in particular") == "other"
    assert label("other") == "Everything Else"
    assert label("not_a_category") == "Everything Else"


# --- the five robotics research subfields ---------------------------------

def test_each_subfield_gets_its_own_bucket():
    assert cat("Sampling-based motion planning for a 7-DoF manipulator") == "robot_planning"
    assert cat("Trajectory optimization for quadruped locomotion") == "robot_optimization"
    assert cat("LTL task specifications for mobile robot navigation") == "robot_formal"
    assert cat("Multi-agent path finding for warehouse robots") == "robot_multiagent"
    assert cat("OpenVLA: an open vision-language-action model") == "robot_ai"


def test_a_subfield_outranks_the_robotics_bucket_it_sits_inside():
    # Every one of these also matches "robotics". Filing on that first would
    # leave the five subfields permanently empty.
    assert cat("MPC for legged robot locomotion") == "robot_planning"
    assert cat("Sim-to-real transfer of dexterous robot manipulation") == "robot_ai"
    # Without a subfield to name, the general bucket is still right.
    assert cat("Boston Dynamics teaches Atlas to fold laundry", "theverge") == "robotics"


def test_overloaded_words_need_a_robot_in_the_room():
    for title, expected in [
        ("Optimization of our query planner", "research"),
        ("A multi-agent LLM framework", "devtools"),
        ("Formal verification of a distributed consensus protocol", "research"),
        ("Foundation models are getting cheaper to serve", "ai_ml"),
    ]:
        assert cat(title) == expected, title
    # The same words, with a robot attached.
    assert cat("Optimization of a robot arm's reaching trajectory") == "robot_optimization"
    assert cat("Multi-agent coordination for a drone fleet") == "robot_multiagent"
    assert cat("Formal verification of a robot's safety controller") == "robot_formal"
    assert cat("Foundation models for robot manipulation") == "robot_ai"


def test_gated_buckets_do_not_fire_without_their_own_vocabulary():
    # A robot story with no research content stays a robot story.
    assert cat("Unitree ships a cheaper humanoid") == "robotics"


# --- the boundary bug the stems used to hide -------------------------------

def test_inflections_and_punctuation_match():
    # `\b(robot|manipulat|vulnerabilit)\b` matched none of these: the
    # trailing boundary fails on the next letter. Same for a term ending in
    # punctuation, where it fails for the opposite reason.
    assert cat("Two robots walk into a factory") == "robotics"
    assert cat("A new dexterous manipulation benchmark") == "robotics"
    assert cat("A critical vulnerability in curl") == "security"
    assert cat("EU regulation of model training") == "policy"
    assert cat("Specifying a protocol in TLA+") == "research"


def test_a_stem_does_not_swallow_an_unrelated_word():
    assert cat("Meta bans a developer account") != "robotics"
    assert cat("Market manipulation probe at an exchange") != "robotics"
