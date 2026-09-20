from tech_news_curator.normalize import (
    canonical_url, cluster_key, same_story, title_key, title_tokens,
)


def test_tracking_parameters_are_dropped_but_content_ones_kept():
    assert canonical_url("https://ex.com/a?utm_source=hn&fbclid=9&id=7") == "https://ex.com/a?id=7"
    # `v` selects the video; dropping it would collapse YouTube to one story.
    assert canonical_url("https://youtube.com/watch?v=abc") == "https://youtube.com/watch?v=abc"


def test_host_and_path_noise_is_normalized():
    same = {
        canonical_url("http://www.example.com/story/"),
        canonical_url("https://example.com/story"),
        canonical_url("https://m.example.com/story/amp/"),
        canonical_url("https://example.com:443/story#section-2"),
    }
    assert len(same) == 1


def test_leading_amp_segment_is_stripped():
    assert canonical_url("https://www.cnbc.com/amp/2026/09/18/x.html") == \
        "https://cnbc.com/2026/09/18/x.html"


def test_unusable_urls_return_empty_so_callers_fall_back_to_the_title():
    assert canonical_url("") == ""
    assert canonical_url("javascript:alert(1)") == ""
    assert canonical_url("not a url") == ""


def test_title_tokens_drop_aggregator_prefixes_and_stopwords():
    assert title_tokens("Show HN: A new database (2024)") == ["database"]
    assert title_key("The rise of the machines in 2026") == "rise machines 2026"


def test_cluster_key_prefers_the_url_and_namespaces_the_fallback():
    assert cluster_key("https://ex.com/a", "Title").startswith("u:")
    assert cluster_key("", "A long enough title").startswith("t:")
    # An empty title with no URL is unclusterable rather than a shared "".
    assert cluster_key("", "") == ""


def test_same_story_needs_containment_not_a_shared_prefix():
    a = {"title": "OpenAI releases GPT-5"}
    b = {"title": "OpenAI releases GPT-5 to all paid tiers"}
    c = {"title": "OpenAI hires a new CFO"}
    assert same_story(a, b)
    assert not same_story(a, c)


def test_same_story_ignores_titles_too_short_to_be_evidence():
    assert not same_story({"title": "Rust 2.0"}, {"title": "Rust 2.0 is out today finally"})
