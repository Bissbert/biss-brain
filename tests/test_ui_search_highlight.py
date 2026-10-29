"""Tests for the server-side _highlight() function (F-008).

Verifies HTML escaping, multi-word highlighting, edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Import directly from the router module
from ui.routers.search import _highlight


class TestHighlightHtmlEscaping:
    def test_escapes_html_entities(self) -> None:
        text = "<script>alert('xss')</script>"
        result = _highlight(text, "")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_ampersand(self) -> None:
        result = _highlight("foo & bar", "")
        assert "&amp;" in result

    def test_escapes_quotes(self) -> None:
        result = _highlight('say "hello"', "")
        assert "&quot;" in result or "&#x27;" in result or "&#34;" in result or '"hello"' not in result

    def test_no_query_returns_escaped_text(self) -> None:
        result = _highlight("plain text", "")
        assert result == "plain text"

    def test_max_len_truncates(self) -> None:
        long_text = "a" * 1000
        result = _highlight(long_text, "")
        assert len(result) == 400

    def test_max_len_custom(self) -> None:
        result = _highlight("hello world", "", max_len=5)
        assert result == "hello"


class TestHighlightTermWrapping:
    def test_single_term_wrapped(self) -> None:
        result = _highlight("the brain stores memories", "brain")
        assert "<mark>brain</mark>" in result

    def test_case_insensitive(self) -> None:
        result = _highlight("The Brain is important", "brain")
        assert "<mark>" in result.lower()
        assert "brain" in result.lower()

    def test_multi_word_query(self) -> None:
        result = _highlight("the brain stores memories efficiently", "brain memories")
        assert "<mark>brain</mark>" in result
        assert "<mark>memories</mark>" in result

    def test_short_terms_skipped(self) -> None:
        # Single-char terms should be skipped to avoid noise
        result = _highlight("a b c hello", "a b")
        # 'a' and 'b' are single chars, no marks expected
        assert "<mark>" not in result

    def test_two_char_term_included(self) -> None:
        result = _highlight("go faster", "go")
        assert "<mark>go</mark>" in result

    def test_partial_word_match(self) -> None:
        # regex matches substrings too
        result = _highlight("learning and learned", "learn")
        assert "<mark>" in result

    def test_special_regex_chars_in_query_escaped(self) -> None:
        # Query with regex metacharacters should not raise
        result = _highlight("price is $100.00 total", "$100")
        # Should not raise; $ is re.escape-d
        assert isinstance(result, str)

    def test_empty_query_no_marks(self) -> None:
        result = _highlight("some text", "")
        assert "<mark>" not in result

    def test_whitespace_only_query(self) -> None:
        result = _highlight("some text", "   ")
        assert "<mark>" not in result

    def test_html_in_text_escaped_before_mark(self) -> None:
        # The text <b>bold</b> should be escaped; marks should still appear around query
        result = _highlight("<b>bold decision</b>", "decision")
        assert "&lt;b&gt;" in result
        assert "<mark>decision</mark>" in result

    def test_mark_does_not_double_escape(self) -> None:
        # The <mark> tags themselves must not be escaped
        result = _highlight("decision point", "decision")
        assert "<mark>decision</mark>" in result
        assert "&lt;mark&gt;" not in result


class TestHighlightTruncationWithTerms:
    def test_term_beyond_max_len_not_highlighted(self) -> None:
        # Text is 10 chars; max_len=5; term only exists after position 5
        result = _highlight("helloworld", "world", max_len=5)
        assert "<mark>" not in result

    def test_term_within_max_len_highlighted(self) -> None:
        result = _highlight("hello world extra", "hello", max_len=11)
        assert "<mark>hello</mark>" in result
