"""Parser tests against captured real responses.

The fixtures are genuine API output, trimmed - the point is to assert
against the shapes these services actually return rather than the shapes we
imagined. None of these tests touch the network.
"""
import json

from tests.conftest import fixture

from tech_news_curator.fetchers.arxiv import ArxivFetcher
from tech_news_curator.fetchers.bluesky import BlueskyFetcher, _slug_title
from tech_news_curator.fetchers.github import GitHubTrendingFetcher
from tech_news_curator.fetchers.hackernews import HackerNewsFetcher
from tech_news_curator.fetchers.lobsters import LobstersFetcher
from tech_news_curator.fetchers.reddit import RedditFetcher, _outbound
from tech_news_curator.fetchers.rss import RSSFetcher


def test_hackernews_reads_points_comments_and_the_outbound_link():
    items = HackerNewsFetcher().parse(json.loads(fixture("hn_search.json")))
    assert items
    first = items[0]
    assert first.source == "hackernews"
    assert first.points > 0 and first.metric == "points"
    assert first.discussion_url.startswith("https://news.ycombinator.com/item?id=")
    assert first.rank == 1


def test_hackernews_text_posts_fall_back_to_their_own_thread():
    data = {"hits": [{"objectID": "1", "title": "Ask HN: anything?", "url": None,
                      "points": 30, "num_comments": 5, "created_at_i": 1758300000}]}
    item = HackerNewsFetcher().parse(data)[0]
    assert item.url == item.discussion_url


def test_hackernews_skips_hits_with_no_title():
    data = {"hits": [{"objectID": "1", "title": None, "story_title": None}]}
    assert HackerNewsFetcher().parse(data) == []


def test_lobsters_keeps_article_and_thread_apart():
    items = LobstersFetcher().parse(json.loads(fixture("lobsters_hottest.json")))
    assert items
    assert items[0].discussion_url.startswith("https://lobste.rs/s/")
    assert items[0].points >= 0
    assert items[0].tags


def test_github_titles_carry_the_description_so_the_row_means_something():
    items = GitHubTrendingFetcher().parse(json.loads(fixture("github_search.json")))
    assert items
    assert " - " in items[0].title
    assert items[0].metric == "stars" and items[0].points > 0


def test_arxiv_flattens_wrapped_titles_and_strips_the_version_suffix():
    items = ArxivFetcher().parse(fixture("arxiv.xml"))
    assert items
    assert "\n" not in items[0].title
    assert not items[0].source_id.endswith(("v1", "v2"))
    assert items[0].metric == "", "arXiv publishes no popularity signal"


def test_reddit_rss_recovers_the_article_url_from_the_link_anchor():
    items = RedditFetcher().parse_rss(fixture("reddit_top.xml"))
    assert items
    first = items[0]
    assert first.discussion_url.startswith("https://www.reddit.com/r/")
    assert not first.url.startswith("https://www.reddit.com/r/"), \
        "the outbound article, not the thread, is what clusters across sources"
    assert first.metric == "rank" and first.rank == 1


def test_outbound_ignores_a_body_with_no_link_anchor():
    assert _outbound("submitted by someone") == ""
    assert _outbound('<a href="https://ex.com/a">[link]</a>') == "https://ex.com/a"


def test_reddit_is_unauthenticated_without_both_credentials():
    assert not RedditFetcher(client_id="x").authenticated
    assert RedditFetcher(client_id="x", client_secret="y").authenticated


def test_reddit_oauth_json_separates_self_posts_from_links():
    data = {"data": {"children": [
        {"data": {"id": "a", "title": "A link post", "url": "https://ex.com/x",
                  "permalink": "/r/p/comments/a/", "score": 10, "num_comments": 2,
                  "is_self": False, "subreddit": "programming"}},
        {"data": {"id": "b", "title": "A text post", "url": "https://reddit.com/r/p/comments/b/",
                  "permalink": "/r/p/comments/b/", "score": 5, "num_comments": 1,
                  "is_self": True, "subreddit": "programming"}},
    ]}}
    link, text = RedditFetcher().parse(data)
    assert link.url == "https://ex.com/x"
    assert text.url == text.discussion_url, "a self-post is its own thread, not an outbound link"
    assert link.tags == ["r/programming"]


def test_rss_handles_rss2_cdata_dates_images_and_skips_linkless_items():
    items = RSSFetcher("ex", "http://x").parse(fixture("rss2.xml"))
    assert [i.title for i in items] == ["A new SMT solver lands", "Second story"]
    first = items[0]
    assert first.published.year == 2026
    assert first.author == "Jane Roe"
    assert first.tags == ["verification", "tools"]
    assert first.image == "https://cdn.example.com/a.png"
    assert "<b>" not in first.summary and "things" in first.summary
    assert items[1].published is None, "an unparseable date is None, not a wrong date"
    assert items[1].image == "https://cdn.example.com/b.jpg"


def test_rss_prefers_the_atom_alternate_link_over_replies():
    items = RSSFetcher("ex", "http://x").parse(fixture("atom.xml"))
    assert items[0].url == "https://example.org/control"
    assert items[0].author == "Alex Doe"
    assert items[0].tags == ["robotics"]


def test_rss_source_tags_are_merged_with_the_feeds_own():
    items = RSSFetcher("ex", "http://x", tags=["mine"]).parse(fixture("atom.xml"))
    assert items[0].tags == ["mine", "robotics"]


def test_bluesky_requires_a_link_and_takes_the_headline_from_the_slug():
    post = {
        "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
        "author": {"handle": "someone.bsky.social"},
        "record": {"text": "this", "createdAt": "2026-09-19T10:00:00Z"},
        "embed": {"external": {"uri": "https://ex.com/blog/a-new-robot-arm"}},
        "likeCount": 100, "repostCount": 10, "replyCount": 3,
    }
    item = BlueskyFetcher(min_likes=1).parse({"posts": [post]}, "robotics")[0]
    assert item.url == "https://ex.com/blog/a-new-robot-arm"
    assert item.title.startswith("A new robot arm"), "a bare 'this' is not a headline"
    assert item.points == 120, "reposts count double; they cost more than a like"
    assert item.discussion_url == "https://bsky.app/profile/someone.bsky.social/post/xyz"


def test_bluesky_drops_linkless_posts_and_quiet_ones():
    chatter = {"uri": "at://a/b/c", "author": {"handle": "x"},
               "record": {"text": "lol AI"}, "likeCount": 900}
    quiet = {"uri": "at://a/b/d", "author": {"handle": "x"},
             "record": {"text": "real news"}, "likeCount": 2,
             "embed": {"external": {"uri": "https://ex.com/a"}}}
    assert BlueskyFetcher(min_likes=25).parse({"posts": [chatter, quiet]}) == []


def test_slug_title_ignores_slugs_too_thin_to_be_a_headline():
    assert _slug_title("https://ex.com/123456") == ""
    assert _slug_title("https://ex.com/a/b/the-big-story") == "The big story"
