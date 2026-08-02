"""Tests for the filters that decide which clusters are worth paying to extract.

Each of these encodes something a 300 cluster run actually got wrong, so they
are regression tests rather than speculation.
"""

from __future__ import annotations

from precedent.extract.cluster import Cluster, ClusterComment


def comment(cid, pr, author, distance, body="a comment long enough to matter"):
    return ClusterComment(
        id=cid,
        pr_number=pr,
        author=author,
        file_path=None,
        created_at=None,
        body=body,
        distance=distance,
    )


def cluster(*comments, seed="s"):
    return Cluster(seed_id=seed, comments=list(comments))


class TestMeanDistance:
    def test_excludes_the_seed(self):
        # The seed comes back at distance zero. Counting it made every cluster
        # look tighter than it was, by roughly 1/k.
        c = cluster(
            comment("s", 1, "a", 0.0),
            comment("b", 2, "b", 0.8),
            comment("c", 3, "c", 0.8),
        )
        assert c.mean_distance == 0.8

    def test_seed_only_cluster_is_zero(self):
        assert cluster(comment("s", 1, "a", 0.0)).mean_distance == 0.0


class TestRejection:
    def test_accepts_a_real_convention(self):
        c = cluster(
            comment("s", 1, "maintainer_a", 0.0),
            comment("b", 2, "maintainer_b", 0.70),
            comment("c", 3, "maintainer_c", 0.75),
        )
        assert c.rejection_reason() is None
        assert c.is_worth_extracting()

    def test_rejects_one_conversation(self):
        # Three comments, one pull request: restated, not repeated.
        c = cluster(
            comment("s", 7, "a", 0.0),
            comment("b", 7, "b", 0.6),
            comment("c", 7, "c", 0.7),
        )
        assert c.rejection_reason() == "too few distinct pull requests"

    def test_rejects_a_single_voice(self):
        # One maintainer posting the same guidance on eight PRs is an opinion,
        # and in practice this catches canned replies.
        c = cluster(*(comment(str(i), i, "same_person", 0.5) for i in range(8)), seed="0")
        assert c.rejection_reason() == "a single author"

    def test_rejects_duplicated_boilerplate(self):
        # The pull request template pasted into eight PRs looks like
        # overwhelming evidence. Its comments are textually identical, so the
        # distances collapse to nearly zero.
        c = cluster(
            *(comment(str(i), i, f"person_{i}", 0.01) for i in range(8)),
            seed="0",
        )
        assert c.rejection_reason() == "duplicated text, not a convention"

    def test_distinct_pull_requests_checked_before_authors(self):
        # Ordering matters only for the message, but a confusing reason costs
        # time when reading a run's summary.
        c = cluster(comment("s", 1, "a", 0.0), comment("b", 1, "a", 0.5))
        assert c.rejection_reason() == "too few distinct pull requests"

    def test_thresholds_are_overridable(self):
        c = cluster(
            comment("s", 1, "a", 0.0),
            comment("b", 2, "b", 0.5),
        )
        assert c.rejection_reason() == "too few distinct pull requests"
        assert c.rejection_reason(min_distinct_prs=2) is None
