"""Tests for embedding input truncation.

This is guarding against a failure that actually happened. The original cap was
on characters, chosen by assuming four characters per token. Diff hunks and
code in the pandas corpus run as dense as 2.18 characters per token, so a
24,000 character input reached 11,002 tokens against an 8,192 token limit, and
the API rejected the entire batch with a 400.
"""

from __future__ import annotations

import tiktoken

from precedent.embed.provider import MAX_TOKENS, truncate_for_embedding

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(value: str) -> int:
    return len(ENCODING.encode(value, disallowed_special=()))


class TestTruncation:
    def test_short_text_is_untouched(self):
        # Stability matters: unchanged text keeps its content hash, which is
        # what lets the embedding cache survive a change to this function.
        body = "Could you add a whatsnew note for this change?"
        assert truncate_for_embedding(body) == body

    def test_long_prose_is_cut_to_the_limit(self):
        body = "the quick brown fox jumps over the lazy dog. " * 3000
        assert token_count(body) > MAX_TOKENS
        assert token_count(truncate_for_embedding(body)) <= MAX_TOKENS

    def test_dense_code_is_cut_to_the_limit(self):
        # Roughly the shape that broke the backfill: punctuation-heavy text
        # that tokenizes far worse than prose.
        body = "@@ -1,4 +1,4 @@\n-    x = df.loc[:, ['a','b']]\n+    x = df[['a','b']]\n" * 2000
        assert len(body) / token_count(body) < 3, "fixture is not dense enough to be a real test"
        assert token_count(truncate_for_embedding(body)) <= MAX_TOKENS

    def test_result_stays_under_the_api_hard_limit(self):
        body = "x = {'a': 1, 'b': [2, 3]}  # comment\n" * 5000
        assert token_count(truncate_for_embedding(body)) <= 8192

    def test_empty_string(self):
        assert truncate_for_embedding("") == ""

    def test_unicode_survives_the_round_trip(self):
        # Truncation decodes back from tokens, so it must not split a character.
        body = "naïve café résumé 日本語 " * 5000
        result = truncate_for_embedding(body)
        assert isinstance(result, str)
        assert token_count(result) <= MAX_TOKENS
        result.encode("utf-8")
