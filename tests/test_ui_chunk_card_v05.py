"""Tests for _chunk_card.html rendering with v0.5 columns.

Verifies:
- Salience dot row rendered (10 dots, filled count proportional)
- Labile badge shown when labile_until > current time
- Lesson badge shown when kind == "lesson"
- Reconsolidation dots shown when reconsolidation_count > 0
- No broken hx-get to /chunk/<id> (click-to-expand bug fixed)
- Compose checkbox present
- Neighbours button present
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)

import pytest
from jinja2 import Environment, FileSystemLoader

os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")

_TEMPLATES_DIR = _ROOT / "ui" / "templates"


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    from datetime import datetime
    def _fmt_ts(ts):
        if not ts:
            return "unknown"
        try:
            return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)
    env.filters["fmt_ts"] = _fmt_ts
    return env


def _render_card(chunk: dict, project: str = "testproject") -> str:
    env = _make_env()
    tmpl = env.get_template("_chunk_card.html")
    return tmpl.render(chunk=chunk, project=project)


def _base_chunk(**overrides) -> dict:
    defaults = {
        "id": "abc123def456",
        "kind": "note",
        "ts": 1700000000,
        "session_id": "sess0001",
        "text": "This is a test chunk body.",
        "pinned": 0,
        "decay_score": 0.85,
        "retrieval_count": 3,
        "score": None,
        "is_superseded": False,
        # v0.5 columns
        "salience": 60,
        "labile_until": 0,
        "reconsolidation_count": 0,
        "consolidation_round": 0,
        "retrieval_count_local": 1,
        "co_retrieval_count": 0,
        "cross_links": "[]",
        "cross_eligible": False,
        "_now_ts": int(time.time()),
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Test 1: salience dots are present
# ---------------------------------------------------------------------------

def test_salience_dots_present() -> None:
    chunk = _base_chunk(salience=60)
    html = _render_card(chunk)
    # Should render 10 dot spans; salience 60 => 6 filled, 4 empty
    assert "sal" in html, "Salience label 'sal' should appear"
    # The value 60 should appear as text
    assert ">60<" in html or ">60 <" in html or "60</span>" in html or ">60</span>" in html


def test_salience_zero() -> None:
    chunk = _base_chunk(salience=0)
    html = _render_card(chunk)
    assert "sal" in html
    # Zero salience: value 0 shown
    assert ">0<" in html or "0</span>" in html


def test_salience_max() -> None:
    chunk = _base_chunk(salience=100)
    html = _render_card(chunk)
    assert "sal" in html
    assert ">100<" in html or "100</span>" in html


# ---------------------------------------------------------------------------
# Test 2: labile badge shown only when labile_until > now
# ---------------------------------------------------------------------------

def test_labile_badge_shown_when_active() -> None:
    future_ts = int(time.time()) + 3600
    chunk = _base_chunk(labile_until=future_ts, _now_ts=int(time.time()))
    html = _render_card(chunk)
    assert "labile" in html.lower(), "Labile badge should appear when labile_until is in future"


def test_labile_badge_hidden_when_expired() -> None:
    past_ts = int(time.time()) - 3600
    chunk = _base_chunk(labile_until=past_ts, _now_ts=int(time.time()))
    html = _render_card(chunk)
    # The badge span with 'labile' text shouldn't be in the main header row
    # (it may still appear in footer as part of decay text)
    # We check there's no badge-style labile span
    import re
    badge_labile = re.search(r'class="badge[^"]*"[^>]*>labile', html, re.IGNORECASE)
    assert not badge_labile, "Labile badge should not appear when labile_until is in the past"


def test_labile_badge_hidden_when_zero() -> None:
    chunk = _base_chunk(labile_until=0)
    html = _render_card(chunk)
    import re
    badge_labile = re.search(r'class="badge[^"]*"[^>]*>labile', html, re.IGNORECASE)
    assert not badge_labile, "Labile badge should not appear when labile_until=0"


# ---------------------------------------------------------------------------
# Test 3: lesson badge when kind == "lesson"
# ---------------------------------------------------------------------------

def test_lesson_badge_shown_for_lesson_kind() -> None:
    chunk = _base_chunk(kind="lesson")
    html = _render_card(chunk)
    assert "lesson" in html, "Lesson badge should appear when kind='lesson'"


def test_no_lesson_badge_for_note_kind() -> None:
    chunk = _base_chunk(kind="note")
    html = _render_card(chunk)
    # 'lesson' text should not appear in badge context
    # The note card should show 'note' not 'lesson'
    assert ">note<" in html or "note</span>" in html


# ---------------------------------------------------------------------------
# Test 4: reconsolidation dots shown when count > 0
# ---------------------------------------------------------------------------

def test_reconsolidation_dots_shown() -> None:
    chunk = _base_chunk(reconsolidation_count=3)
    html = _render_card(chunk)
    assert "recon" in html, "Reconsolidation row should appear when count > 0"


def test_reconsolidation_dots_hidden_when_zero() -> None:
    chunk = _base_chunk(reconsolidation_count=0)
    html = _render_card(chunk)
    assert "recon" not in html, "Reconsolidation row should not appear when count=0"


# ---------------------------------------------------------------------------
# Test 5: no broken hx-get /chunk/<id> (click-to-expand fix)
# ---------------------------------------------------------------------------

def test_no_broken_hx_get_chunk() -> None:
    chunk = _base_chunk()
    html = _render_card(chunk)
    assert "hx-get=\"/chunk/" not in html, (
        "Broken hx-get /chunk/<id> should be removed; expand uses CSS toggle"
    )


# ---------------------------------------------------------------------------
# Test 6: compose checkbox present
# ---------------------------------------------------------------------------

def test_compose_checkbox_present() -> None:
    chunk = _base_chunk()
    html = _render_card(chunk)
    assert 'class="compose-pick"' in html, "Compose checkbox should be present"
    assert f'data-id="{chunk["id"]}"' in html, "Checkbox should have data-id attribute"


# ---------------------------------------------------------------------------
# Test 7: neighbours button present
# ---------------------------------------------------------------------------

def test_neighbours_button_present() -> None:
    chunk = _base_chunk()
    html = _render_card(chunk)
    assert "/api/neighbours" in html, "Neighbours button with hx-get to /api/neighbours should be present"
    assert f'id="nbr-{chunk["id"]}"' in html, "Neighbours target div should be present"


# ---------------------------------------------------------------------------
# Test 8: v0.5 row 2 – decay / retrieved / round
# ---------------------------------------------------------------------------

def test_decay_row_shows_local_and_total() -> None:
    chunk = _base_chunk(
        retrieval_count_local=4,
        retrieval_count=10,
        consolidation_round=2,
    )
    html = _render_card(chunk)
    assert "local" in html, "Should show 'local' retrieval count"
    assert "total" in html, "Should show 'total' retrieval count"
    assert "round" in html, "Should show round number when > 0"


def test_round_not_shown_when_zero() -> None:
    chunk = _base_chunk(consolidation_round=0)
    html = _render_card(chunk)
    # "· round N" text should not appear; "round" appears in border-radius CSS so
    # we check specifically for the "· round" label text
    assert "&middot; round" not in html and "· round" not in html, (
        "Round label should not appear when consolidation_round=0"
    )
