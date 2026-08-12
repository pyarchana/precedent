"""Tests for finding the maintainer list, which is easy to lose quietly.

An absent list is not an error. It returns an empty set and authorship checks
fall back to `authorAssociation` alone, which is precisely the defect
`scripts/derive_maintainers.py` exists to correct: on pandas that field
misreports about a third of the corpus, because it reflects permissions held
now rather than when the comment was written.

So the failure mode is a deployment that works, answers, and is wrong about who
speaks for the project, with one log line to show for it. A deployment did ship
without the file. These tests are about the path, not the parsing.
"""

from __future__ import annotations

import pytest
import yaml

from precedent.transform.maintainers import (
    CANDIDATE_PATHS,
    find_maintainer_list,
    load_maintainers,
)


class TestFindingTheList:
    def test_the_checked_in_list_is_found_from_a_checkout(self):
        assert find_maintainer_list().is_file()

    def test_both_layouts_are_looked_for(self):
        # A checkout nests the package under `src/`; the Lambda package does
        # not, so `REPO_ROOT` resolves to `/` there and the list has to be
        # looked for beside the package instead.
        assert len(CANDIDATE_PATHS) == 2
        assert all(p.name == "maintainers.yaml" for p in CANDIDATE_PATHS)

    def test_a_missing_list_still_names_somewhere_to_put_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "precedent.transform.maintainers.CANDIDATE_PATHS",
            (tmp_path / "nowhere" / "maintainers.yaml",),
        )
        assert find_maintainer_list().name == "maintainers.yaml"


class TestLoading:
    def test_the_real_list_loads(self):
        assert len(load_maintainers("pandas-dev/pandas")) > 0

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        # A fresh repository has no list yet, and classification by association
        # still works. Refusing to start would be worse.
        assert load_maintainers("x/y", tmp_path / "absent.yaml") == frozenset()

    def test_a_list_for_another_repository_is_refused(self, tmp_path):
        # Silently applying pandas' maintainers to another project would
        # corrupt the corpus in a way nothing downstream could detect.
        path = tmp_path / "maintainers.yaml"
        path.write_text(
            yaml.safe_dump({"repo": "pandas-dev/pandas", "maintainers": [{"login": "jreback"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="pandas-dev/pandas"):
            load_maintainers("polars/polars", path)

    def test_logins_are_lowercased(self, tmp_path):
        path = tmp_path / "maintainers.yaml"
        path.write_text(
            yaml.safe_dump({"repo": "x/y", "maintainers": [{"login": "JReback"}]}),
            encoding="utf-8",
        )
        assert load_maintainers("x/y", path) == frozenset({"jreback"})
