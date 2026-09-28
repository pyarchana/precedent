"""Tests for request instrumentation.

The property that matters is negative: none of this may change what a request
does, or raise on its own.
"""

from __future__ import annotations

import json
import logging

import pytest

from precedent import obs


@pytest.fixture(autouse=True)
def clean():
    obs.reset()
    yield
    obs.reset()


class TestItNeverBreaksTheRequest:
    def test_an_exception_passes_through_untouched(self):
        boom = ValueError("the database went away")
        with pytest.raises(ValueError) as caught, obs.request("ask"):
            raise boom
        assert caught.value is boom

    def test_a_failing_stage_is_still_timed_and_still_raises(self):
        with pytest.raises(RuntimeError), obs.request("ask") as rec, rec.stage("recall"):
            raise RuntimeError("no")
        assert "recall" in rec.stages

    def test_a_field_that_cannot_be_serialised_does_not_raise(self, caplog):
        class Awkward:
            def __repr__(self):
                raise RuntimeError("not even repr")

        with caplog.at_level(logging.INFO), obs.request("ask") as rec:
            rec.note(thing=Awkward())
        # The record exists; formatting it is the formatter's problem, and the
        # formatter is tested separately for exactly this.
        assert caplog.records

    def test_the_formatter_falls_back_rather_than_raising(self):
        class Awkward:
            def __repr__(self):
                raise RuntimeError("not even repr")

        record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)
        record.thing = Awkward()
        line = obs.JsonFormatter().format(record)
        assert json.loads(line)["msg"] == "hello"


class TestWhatItRecords:
    def test_an_exception_is_labelled_by_its_type(self):
        with pytest.raises(KeyError), obs.request("ask"):
            raise KeyError("k")
        assert obs.snapshot()["outcomes"]["ask"] == {"error:KeyError": 1}

    def test_outcomes_are_counted_separately(self):
        # The point of the whole exercise: a refusal and a parse failure used to
        # be the same line in the log.
        for name in ("declined", "declined", "citations_unverified"):
            with obs.request("ask") as rec:
                rec.outcome(name)
        assert obs.snapshot()["outcomes"]["ask"] == {"declined": 2, "citations_unverified": 1}

    def test_a_clean_request_is_ok(self):
        with obs.request("ask"):
            pass
        assert obs.snapshot()["outcomes"]["ask"] == {"ok": 1}

    def test_stages_land_on_the_log_line(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="precedent.request"),
            obs.request("ask", question_len=12) as rec,
        ):
            with rec.stage("recall"):
                pass
            rec.note(rules=3)

        record = caplog.records[-1]
        assert record.route == "ask"
        assert record.question_len == 12
        assert record.rules == 3
        assert "recall" in record.stages
        assert record.total_ms >= 0

    def test_the_line_is_valid_json_with_the_fields_merged(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="precedent.request"),
            obs.request("webhook.review") as rec,
        ):
            rec.outcome("commented")
            rec.note(pr=7)

        parsed = json.loads(obs.JsonFormatter().format(caplog.records[-1]))
        assert parsed["route"] == "webhook.review"
        assert parsed["outcome"] == "commented"
        assert parsed["pr"] == 7
        assert parsed["level"] == "INFO"


class TestTheSnapshotDoesNotOverclaim:
    def test_it_says_its_scope_is_one_process(self):
        # Lambda runs several execution environments and recycles them, so a
        # number here is not a total. Saying so is the difference between a
        # useful glance and a misleading dashboard.
        assert "execution environment" in obs.snapshot()["scope"]

    def test_latency_is_reported_per_route(self):
        with obs.request("ask"):
            pass
        with obs.request("rules"):
            pass
        latency = obs.snapshot()["latency"]
        assert set(latency) == {"ask", "rules"}
        assert latency["ask"]["n"] == 1

    def test_the_sample_is_bounded(self):
        # An endpoint under load must not grow this without limit.
        for _ in range(obs.SAMPLE_SIZE + 50):
            with obs.request("ask"):
                pass
        assert obs.snapshot()["latency"]["ask"]["n"] == obs.SAMPLE_SIZE
        assert obs.snapshot()["outcomes"]["ask"]["ok"] == obs.SAMPLE_SIZE + 50


class TestLoggingSetup:
    def test_configuring_twice_does_not_double_the_handlers(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            obs.configure_logging()
            after_one = len([h for h in root.handlers if getattr(h, "_precedent", False)])
            obs.configure_logging()
            after_two = len([h for h in root.handlers if getattr(h, "_precedent", False)])
            assert after_one == after_two == 1
        finally:
            root.handlers = original

    def test_lambda_gets_json_without_being_asked(self, monkeypatch):
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "precedent-api")
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            obs.configure_logging()
            installed = [h for h in root.handlers if getattr(h, "_precedent", False)]
            assert isinstance(installed[0].formatter, obs.JsonFormatter)
        finally:
            root.handlers = original
