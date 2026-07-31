"""Tests for the VECTOR wire format codec.

The driver hands these columns back as text, so every embedding read and write
crosses this boundary. A silent malformation here would not raise, it would
just make every distance meaningless.
"""

from __future__ import annotations

import pytest

from precedent.embed.vector import decode, encode


class TestEncode:
    def test_renders_a_bracketed_list(self):
        assert encode([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"

    def test_accepts_integers(self):
        assert encode([1, 2, 3]) == "[1.0,2.0,3.0]"

    def test_empty_vector(self):
        assert encode([]) == "[]"

    def test_keeps_full_precision(self):
        value = 0.1234567890123456
        assert str(value) in encode([value])


class TestDecode:
    def test_parses_what_the_driver_returns(self):
        assert decode("[1,2.5,-3]") == [1.0, 2.5, -3.0]

    def test_none_stays_none(self):
        # An unembedded row, which is different from a zero-length vector.
        assert decode(None) is None

    def test_empty_vector(self):
        assert decode("[]") == []

    def test_tolerates_surrounding_whitespace(self):
        assert decode("  [1,2]  ") == [1.0, 2.0]


class TestRoundTrip:
    @pytest.mark.parametrize(
        "values",
        [
            [0.0],
            [1.0, -1.0],
            [1e-8, 1e8],
            [0.1, 0.2, 0.3],
            list(range(100)),
        ],
    )
    def test_survives_a_round_trip(self, values):
        assert decode(encode(values)) == [float(v) for v in values]

    def test_dimension_is_preserved(self):
        values = [0.5] * 1536
        assert len(decode(encode(values))) == 1536
