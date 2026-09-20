from tech_news_curator.popularity import (
    percentile_ranks, popularity_scores, propagate,
)


def row(item_id, source, points=0, comments=0, metric="points", rank=0, cluster="u:x"):
    return {"id": item_id, "source": source, "points": points, "comments": comments,
            "metric": metric, "rank": rank, "cluster_key": cluster}


def test_percentile_ranks_share_the_midpoint_on_ties():
    assert percentile_ranks([5, 5, 5]) == [0.5, 0.5, 0.5]
    assert percentile_ranks([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert percentile_ranks([7]) == [1.0]


def test_scores_are_relative_to_the_items_own_source():
    # 200 GitHub stars is unremarkable; 200 Hacker News points is a big day.
    rows = [row(1, "hackernews", 200), row(2, "hackernews", 10), row(3, "hackernews", 5),
            row(4, "github", 200, metric="stars"), row(5, "github", 9000, metric="stars"),
            row(6, "github", 12000, metric="stars")]
    scores = popularity_scores(rows)
    assert scores[1][0] > scores[4][0], "a source's own distribution sets the scale"


def test_sources_without_counts_get_no_number_rather_than_a_made_up_one():
    rows = [row(1, "theverge", metric=""), row(2, "arxiv", metric="")]
    assert popularity_scores(rows) == {}


def test_rank_only_sources_are_scored_from_their_ordering():
    rows = [row(i, "reddit", metric="rank", rank=i, cluster=f"u:{i}") for i in range(1, 6)]
    scores = popularity_scores(rows)
    assert len(scores) == 5
    assert scores[1][0] > scores[5][0]
    assert scores[1][1] == "#1 on reddit"


def test_comments_count_but_at_a_discount():
    # Measured against a realistic spread: in a two-item source both items
    # are extremes of their own distribution and nothing is observable.
    field = [row(i, "hackernews", points=i * 40, cluster=f"u:{i}") for i in range(2, 12)]
    quiet = popularity_scores([row(1, "hackernews", 200)] + field)
    loud = popularity_scores([row(1, "hackernews", 200, comments=400)] + field)
    assert loud[1][0] > quiet[1][0]
    # ...but a 400-comment argument is not worth 400 more upvotes.
    upvoted = popularity_scores([row(1, "hackernews", 600)] + field)
    assert loud[1][0] < upvoted[1][0]


def test_reason_does_not_claim_a_standing_below_the_median():
    rows = [row(1, "hackernews", 5), row(2, "hackernews", 500), row(3, "hackernews", 900)]
    assert "top" not in popularity_scores(rows)[1][1], "'top 100%' is not a standing"
    assert "top" in popularity_scores(rows)[3][1]


def test_propagate_shares_attention_across_copies_of_one_story():
    rows = [row(1, "hackernews", 900, cluster="u:same"),
            row(2, "theverge", metric="", cluster="u:same"),
            row(3, "arstechnica", metric="", cluster="u:other")]
    scores = popularity_scores(rows)
    shared = propagate(rows, scores)
    assert shared[2][0] == shared[1][0], "the publisher's copy inherits the aggregator's attention"
    assert "via hackernews" in shared[2][1], "an inherited number must say where it came from"
    assert 3 not in shared, "an unrelated story inherits nothing"


def test_propagate_never_lowers_a_score_and_leaves_the_input_alone():
    rows = [row(1, "hackernews", 900, cluster="u:same"),
            row(2, "reddit", metric="rank", rank=40, cluster="u:same")]
    scores = popularity_scores(rows)
    before = dict(scores)
    shared = propagate(rows, scores)
    assert scores == before
    assert shared[1][0] >= before[1][0] and shared[2][0] >= before[2][0]
