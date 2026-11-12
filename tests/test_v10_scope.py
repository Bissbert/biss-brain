"""Stage 1: a memory's scope (project | general) from the distiller and from arcs."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import transcript as T  # noqa: E402
from test_v07_segment import _batch, _ev, _line  # noqa: E402


def _out(*bullets) -> str:
    return json.dumps({"bullets": list(bullets)})


def test_general_bullets_get_their_own_group_only_when_trusted():
    raw = _out(
        {"source": "user", "section": "Decisions", "text": "Use ruff here.", "salience": 8},
        {"source": "claude", "text": "uv run ignores VIRTUAL_ENV without --active.", "salience": 6, "scope": "general"},
        {"source": "web", "text": "pip 25 drops py3.8.", "salience": 9, "scope": "GENERAL"},
        {"source": "user", "text": "Odd scope.", "salience": 3, "scope": "everywhere"},
    )
    groups, sal = T.split_output(raw, ["user", "assistant", "web"])
    assert groups["general"] == [("assistant", "- uv run ignores VIRTUAL_ENV without --active.")]
    assert ("user", "- Odd scope.") in groups["trusted"]  # unknown scope reads as project
    assert groups["low"] == [("web", "- pip 25 drops py3.8.")]  # web never goes general
    assert sal["general"] == 6
    # no general bullet: the group is absent (callers and older tests see two groups)
    assert set(T.split_output(_out({"source": "user", "text": "x", "salience": 5}), ["user"])[0]) == {"trusted", "low"}


def test_distill_prompts_document_scope():
    import distill

    for k in ("stop", "subagent_stop", "pre_compact"):
        tpl = distill._load_prompt(k)
        assert '"scope": "general"' in tpl and distill.placeholders(tpl) == {"text"}


def test_flush_stores_general_bullets_as_their_own_chunk(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "use ruff") + _line("assistant", "uv needs --active"))
    raw = _out({"source": "user", "text": "The user wants ruff in this repo.", "salience": 8},
               {"source": "claude", "text": "uv run ignores VIRTUAL_ENV without --active.", "salience": 6,
                "scope": "general"})
    seen, _ = _batch(tmp_path, monkeypatch, [_ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))],
                     summary=raw)
    meta = common.open_meta(str(cwd))
    rows = sorted(tuple(r) for r in meta.execute("SELECT scope, salience FROM chunks"))
    assert rows == [("general", 6), ("project", 8)]


def test_schema_adds_scope_with_project_default(tmp_path):
    c = sqlite3.connect(str(tmp_path / "m.sqlite"))
    c.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, project TEXT, session_id TEXT, agent_id TEXT, kind TEXT,"
              " ts INTEGER, content_hash TEXT, superseded_by TEXT DEFAULT '')")
    c.execute("INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES ('a','p','stop',1,'h')")
    common.ensure_schema(c)
    assert c.execute("SELECT scope FROM chunks").fetchone()[0] == "project"


def test_split_lessons():
    import dream

    body, lessons = dream.split_lessons(
        "- Built the importer.\n- Touched a.py.\nLESSON: sqlite  ATTACH needs a path,\n not a URI.\n"
        "- lesson: pyarrow set_column keeps the schema\nLESSON:\nLESSON: a\nLESSON: b\nLESSON: c")
    assert body == "- Built the importer.\n- Touched a.py.\n not a URI."
    assert lessons == ["sqlite ATTACH needs a path,", "pyarrow set_column keeps the schema", "a"]
    assert dream.split_lessons("- only bullets") == ("- only bullets", [])


def test_arc_lessons_become_general_lesson_chunks(tmp_path, monkeypatch):
    import uuid

    from test_dream_stage2 import _insert_chunk_direct, _reset_modules, _write_test_config

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(_write_test_config(tmp_path)))
    monkeypatch.setenv("HOME", str(tmp_path))
    _reset_modules()
    import common as c
    import dream

    cwd = tmp_path / "p"
    cwd.mkdir()
    sid = uuid.uuid4().hex[:16]
    for i in range(3):
        _insert_chunk_direct(str(cwd), uuid.uuid4().hex, f"chunk {i}", [0.1 + i * 0.01] * 768,
                             session_id=sid, ts=1_700_000_000 + i)
    monkeypatch.setattr(dream, "_distill_arc", lambda *a, **k:
                        "- Built x.\nLESSON: git worktree add refuses a branch already checked out.")
    proj, pdir = c.project_key(str(cwd)), c.project_dir(str(cwd))
    assert dream._stage2a_arcs(pdir, proj, c.load_config(), dry_run=False) == 1
    meta = c.open_meta(str(cwd))
    rows = {r[0]: (r[1], r[2]) for r in meta.execute("SELECT kind, scope, evidence_ids FROM chunks"
                                                     " WHERE kind IN ('arc', 'lesson')")}
    arc_id = meta.execute("SELECT id FROM chunks WHERE kind = 'arc'").fetchone()[0]
    meta.close()
    assert rows == {"arc": ("project", rows["arc"][1]), "lesson": ("general", arc_id)}
    tbl = c.open_or_create_chunks(pdir / "lancedb", 768)
    texts = {r["kind"]: r["text"] for r in tbl.search().where("kind IN ('arc', 'lesson')").to_list()}
    assert texts == {"arc": "- Built x.", "lesson": "- git worktree add refuses a branch already checked out."}
    # a second pass over the same session builds nothing new
    assert dream._stage2a_arcs(pdir, proj, c.load_config(), dry_run=False) == 0
