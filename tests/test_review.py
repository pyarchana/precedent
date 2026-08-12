"""Tests for commenting on a pull request nobody asked about.

Speaking uninvited is the only path in this system where being right is not
enough. An answer nobody wanted is noise however accurate it is, and a
contributor who learns to scroll past the bot stops reading the one comment that
mattered. So most of what is tested here is the decision to stay quiet.

The other half is that the paths must be able to overrule the vector index. A
rule about tests is irrelevant to a pull request that changes no tests, and no
amount of embedding similarity should be able to argue otherwise.
"""

from __future__ import annotations

from precedent.agent.retrieve import RetrievedRule
from precedent.agent.review import (
    MAX_ANCHOR_SHARE,
    MAX_RULES,
    Applicable,
    anchor_share,
    applies_to,
    classify,
    describe,
    normalise_pattern,
    pattern_matches,
    render,
    select,
)
from precedent.memory.search import SearchHit


def rule(
    statement: str = "Add a whatsnew note.",
    *,
    # Anchored by default, because an unanchored rule is now never selected and
    # a helper that produced one would make every test below vacuous.
    scope: str = "directory",
    pattern: str | None = "pandas/core/",
    confidence: float = 0.8,
    distance: float = 0.5,
    origin: str = "extracted",
    citations: list[SearchHit] | None = None,
) -> RetrievedRule:
    return RetrievedRule(
        id=statement[:8],
        statement=statement,
        rationale=None,
        scope=scope,
        scope_pattern=pattern,
        confidence=confidence,
        evidence_count=3,
        distance=distance,
        origin=origin,
        citations=citations or [],
    )


def hit(pr_number: int, url: str | None = None) -> SearchHit:
    from datetime import UTC, datetime

    return SearchHit(
        id=str(pr_number),
        pr_number=pr_number,
        kind="review_thread",
        body="do the thing",
        author="someone",
        is_maintainer=True,
        file_path=None,
        url=url,
        created_at=datetime.now(UTC),
        distance=0.0,
    )


class TestClassify:
    def test_a_test_file_is_recognised_wherever_it_lives(self):
        assert classify(["pandas/tests/groupby/test_apply.py"]).tests
        assert classify(["pandas/core/test_thing.py"]).tests
        assert classify(["conftest.py"]).tests

    def test_source_changes_are_not_test_changes(self):
        touched = classify(["pandas/core/groupby/groupby.py"])
        assert touched.code and not touched.tests and not touched.docs

    def test_documentation_is_recognised(self):
        assert classify(["doc/source/whatsnew/v3.0.0.rst"]).docs

    def test_directories_are_collected_for_the_query(self):
        touched = classify(["pandas/core/groupby/groupby.py", "pandas/core/frame.py"])
        assert touched.directories == ("pandas/core", "pandas/core/groupby")

    def test_a_root_file_contributes_no_directory(self):
        assert classify(["setup.py"]).directories == ()


class TestTheLikePatterns:
    """The patterns in memory are SQL LIKE, not globs, and this cost everything.

    Nothing asked the extraction prompt for `pandas/tests/%` and nothing
    documented it. It is simply what the model wrote. Matching those with
    fnmatch, where `%` is an ordinary character, meant 68 of the 79 path
    anchored rules matched nothing at all, and the only rules that ever fired
    were the two written `pandas/**/*.py`, which cover 63% of the repository.

    The agent looked like it was working while running entirely on its two least
    specific rules, and the symptom was that it commented on 39 of 40 pull
    requests.
    """

    def test_a_like_wildcard_is_a_wildcard(self):
        assert pattern_matches("pandas/tests/%", ["pandas/tests/groupby/test_apply.py"])
        assert pattern_matches("pandas/core/%", ["pandas/core/frame.py"])

    def test_a_like_wildcard_in_the_middle_works(self):
        assert pattern_matches("pandas/tests/%/conftest.py", ["pandas/tests/io/conftest.py"])

    def test_a_like_pattern_still_discriminates(self):
        assert not pattern_matches("pandas/tests/%", ["pandas/core/frame.py"])
        assert not pattern_matches("doc/source/whatsnew/%", ["pandas/core/frame.py"])

    def test_normalising_leaves_a_real_glob_alone(self):
        assert normalise_pattern("pandas/**/*.py") == "pandas/**/*.py"

    def test_normalising_strips_the_decoration_models_add(self):
        assert normalise_pattern("`pandas/core/%`") == "pandas/core/*"
        assert normalise_pattern("./pandas/core/") == "pandas/core/"

    def test_an_underscore_is_not_treated_as_a_wildcard(self):
        # LIKE's single-character wildcard, but underscores are legitimate in
        # Python filenames far more often than they are wildcards.
        assert not pattern_matches("test_x.py", ["testYx.py"])


class TestAnchorSpecificity:
    """A pattern covering most of the repository is not an anchor.

    The measured distribution has a real gap: 54 rules anchor on under 10% of
    known paths, 25 on over 40%, and nothing sits between. So the exact cutoff
    carries no weight, which is the only reason a cutoff here is defensible.
    """

    CORPUS = (
        "pandas/core/frame.py",
        "pandas/core/series.py",
        "pandas/io/sql.py",
        "pandas/tests/test_a.py",
        "pandas/tests/test_b.py",
        "doc/source/whatsnew/v3.rst",
    )

    def test_a_specific_anchor_covers_little(self):
        assert anchor_share("pandas/core/frame.py", self.CORPUS) < MAX_ANCHOR_SHARE

    def test_a_whole_repository_anchor_covers_nearly_everything(self):
        assert anchor_share("pandas/%", self.CORPUS) > MAX_ANCHOR_SHARE

    def test_the_share_is_a_share(self):
        assert anchor_share("pandas/tests/%", self.CORPUS) == 2 / 6

    def test_an_empty_corpus_does_not_divide_by_zero(self):
        assert anchor_share("pandas/%", ()) == 0.0


class TestPatternMatching:
    def test_a_glob_matches(self):
        assert pattern_matches("pandas/core/*.py", ["pandas/core/frame.py"])

    def test_a_glob_matches_deeper_than_it_was_written(self):
        # Patterns come from a model reading a comment, so "*.pyx" is written
        # without a path even when every .pyx file is three directories down.
        assert pattern_matches("*.pyx", ["pandas/_libs/lib.pyx"])

    def test_a_bare_directory_matches_what_is_under_it(self):
        assert pattern_matches("pandas/core/groupby/", ["pandas/core/groupby/ops.py"])
        assert pattern_matches("pandas/core/groupby", ["pandas/core/groupby/ops.py"])

    def test_a_directory_matches_when_written_without_its_parents(self):
        assert pattern_matches("groupby", ["pandas/core/groupby/ops.py"])

    def test_a_bare_filename_matches(self):
        assert pattern_matches("setup.py", ["setup.py"])
        assert pattern_matches("conftest.py", ["pandas/tests/conftest.py"])

    def test_an_unrelated_path_does_not_match(self):
        assert not pattern_matches("pandas/core/", ["doc/source/index.rst"])

    def test_a_prefix_that_is_not_a_path_boundary_does_not_match(self):
        # "pandas/core" must not match "pandas/coreutils/x.py".
        assert not pattern_matches("pandas/core", ["pandas/coreutils/x.py"])

    def test_a_missing_pattern_matches_nothing(self):
        assert not pattern_matches(None, ["anything.py"])
        assert not pattern_matches("   ", ["anything.py"])


class TestPathsOverrulesSimilarity:
    """The point of separating the two signals.

    A testing rule can embed very close to a pull request about groupby, since
    both talk about groupby. It is still irrelevant if no test changed.
    """

    def test_a_testing_rule_is_irrelevant_without_test_changes(self):
        paths = ["pandas/core/groupby/groupby.py"]
        anchored = rule(scope="testing", pattern="pandas/tests/%")
        assert applies_to(anchored, paths, classify(paths)) is None

    def test_a_testing_rule_applies_once_tests_change(self):
        paths = ["pandas/tests/groupby/test_apply.py"]
        anchored = rule(scope="testing", pattern="pandas/tests/%")
        assert applies_to(anchored, paths, classify(paths))

    def test_a_docs_rule_is_irrelevant_to_a_pure_code_change(self):
        paths = ["pandas/core/frame.py"]
        anchored = rule(scope="docs", pattern="doc/source/%")
        assert applies_to(anchored, paths, classify(paths)) is None

    def test_a_scope_without_a_pattern_is_not_an_anchor(self):
        # "testing" alone means "this pull request changes tests", which is true
        # of nearly every pull request pandas receives. It is the scope-level
        # version of the mistake MAX_ANCHOR_SHARE catches at the pattern level.
        paths = ["pandas/tests/groupby/test_apply.py", "doc/source/whatsnew/v3.rst"]
        touched = classify(paths)
        assert applies_to(rule(scope="testing", pattern=None), paths, touched) is None
        assert applies_to(rule(scope="docs", pattern=None), paths, touched) is None

    def test_a_directory_rule_needs_its_directory(self):
        paths = ["pandas/io/parsers/readers.py"]
        touched = classify(paths)
        assert applies_to(rule(scope="directory", pattern="pandas/core/"), paths, touched) is None
        assert applies_to(rule(scope="directory", pattern="pandas/io/"), paths, touched)

    def test_a_directory_rule_with_no_pattern_applies_to_nothing(self):
        # A scope that promises a path and then does not name one cannot be
        # checked, and letting it through would make it apply everywhere.
        paths = ["pandas/core/frame.py"]
        assert applies_to(rule(scope="directory", pattern=None), paths, classify(paths)) is None

    def test_a_rule_that_applies_to_everything_starts_nothing(self):
        # This is the fix for a version that commented on 38 of 40 real pull
        # requests. A convention scoped repo, style, process or api is true of
        # every pull request ever opened, so it is not evidence that this one
        # needs anything said about it, and distance was measured unable to
        # tell which of them is relevant. See the module docstring.
        paths = ["anything/at/all.py"]
        touched = classify(paths)
        for scope in ("repo", "style", "process", "api"):
            assert applies_to(rule(scope=scope, pattern=None), paths, touched) is None

    def test_the_reason_names_the_file_not_the_pattern(self):
        # What makes the comment defensible: the contributor can check the
        # reason against their own diff. They have never seen the pattern and
        # have no reason to trust it.
        paths = ["pandas/core/frame.py", "doc/source/whatsnew/v3.0.0.rst"]
        reason = applies_to(rule(scope="directory", pattern="doc/source/%"), paths, classify(paths))
        assert "doc/source/whatsnew/v3.0.0.rst" in reason
        assert "%" not in reason


class TestSelection:
    def test_a_weakly_evidenced_rule_is_not_worth_interrupting_for(self):
        paths = ["pandas/core/frame.py"]
        assert select([rule(confidence=0.2)], paths, classify(paths)) == []

    def test_distance_does_not_gate_anything(self):
        # Deliberate. Distance was measured flat across the whole rule set: the
        # nearest and tenth-nearest rule to any pull request differ by about
        # 0.10 however the query is phrased, so a threshold on it draws an
        # arbitrary line. It orders what the paths already admitted, nothing
        # more.
        paths = ["pandas/core/frame.py"]
        assert len(select([rule(distance=9.0)], paths, classify(paths))) == 1

    def test_no_more_than_three_are_selected(self):
        paths = ["pandas/core/frame.py"]
        many = [rule(f"Rule number {i}.", distance=0.1 * i) for i in range(10)]
        assert len(select(many, paths, classify(paths))) == MAX_RULES

    def test_what_a_maintainer_said_outranks_a_closer_inference(self):
        paths = ["pandas/core/frame.py"]
        inferred = rule("Inferred.", distance=0.1, origin="extracted")
        stated = rule("Stated.", distance=0.9, origin="taught")
        chosen = select([inferred, stated], paths, classify(paths))
        assert chosen[0].rule.statement == "Stated."

    def test_the_reason_records_what_let_it_through(self):
        paths = ["pandas/core/frame.py"]
        chosen = select([rule(scope="file", pattern="pandas/core/%")], paths, classify(paths))
        assert "pandas/core/frame.py" in chosen[0].reason


class TestRendering:
    def test_every_rule_appears(self):
        selected = [
            Applicable(rule=rule("Add a whatsnew note."), reason="r"),
            Applicable(rule=rule("Use pytest.raises with match."), reason="r"),
        ]
        body = render(selected, "@precedent")
        assert "Add a whatsnew note." in body
        assert "Use pytest.raises with match." in body

    def test_citations_become_links(self):
        selected = [
            Applicable(
                rule=rule(citations=[hit(1234, "https://github.com/x/y/pull/1234")]), reason="r"
            )
        ]
        assert "[#1234](https://github.com/x/y/pull/1234)" in render(selected, "@precedent")

    def test_a_citation_without_a_url_is_still_named_not_invented(self):
        body = render([Applicable(rule=rule(citations=[hit(99, None)]), reason="r")], "@precedent")
        assert "#99" in body
        assert "](" not in body.split("Established in")[1].split("\n")[0]

    def test_the_same_pull_request_is_not_cited_twice(self):
        selected = [Applicable(rule=rule(citations=[hit(7, "u"), hit(7, "u"), hit(8, "u")]), reason="r")]
        assert render(selected, "@precedent").count("#7") == 1

    def test_a_rule_with_no_evidence_rows_claims_no_source(self):
        body = render([Applicable(rule=rule(citations=[]), reason="r")], "@precedent")
        assert "Established in" not in body

    def test_the_correction_route_is_offered_with_the_configured_trigger(self):
        assert "@memorybot" in render([Applicable(rule=rule(), reason="r")], "@memorybot")


class TestTheQueryPutToMemory:
    def test_the_title_carries_the_intent(self):
        paths = ["pandas/core/groupby/groupby.py"]
        assert "Fix groupby apply" in describe("Fix groupby apply", paths, classify(paths))

    def test_areas_are_directories_not_filenames(self):
        paths = [f"pandas/core/thing_{i}.py" for i in range(20)]
        described = describe("A change", paths, classify(paths))
        assert "pandas/core" in described
        assert "thing_0.py" not in described

    def test_a_missing_title_still_produces_a_query(self):
        paths = ["pandas/core/frame.py"]
        assert describe(None, paths, classify(paths)).strip()
