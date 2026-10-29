"""v0.7 phase 2: provenance model, schema migration, /remember source."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import provenance as P  # noqa: E402


# ── trust model ──────────────────────────────────────────────────────────

def test_make_and_derive_take_least_trusted_input():
    assert P.make("user").row() == ("user", 1.0, '["user"]')
    d = P.derive(["assistant", "web"])
    assert d.source == "distilled" and d.trust == 0.3 and d.taint == ("assistant", "web")
    assert P.derive([]).taint == ("imported",)
    c = P.combine([P.make("user"), P.derive(["assistant", "file"])], source="arc")
    assert c.source == "arc" and c.trust == 0.7 and c.taint == ("assistant", "file", "user")


def test_trust_overrides_from_config():
    cfg = {"provenance": {"trust": {"web": 0.1}}}
    assert P.make("web", cfg).trust == 0.1
    assert P.make("user", cfg).trust == 1.0


def test_make_rejects_derived_source():
    try:
        P.make("distilled")
    except ValueError:
        return
    raise AssertionError("distilled is not a raw source")


def test_tool_source_unknown_is_tool_never_file():
    assert P.tool_source("WebFetch") == P.tool_source("WebSearch") == "web"
    assert P.tool_source("mcp__some_server__fetch") == "web"
    assert P.tool_source("Read") == "file"
    assert P.tool_source("Bash") == P.tool_source("SomeNewTool") == P.tool_source(None) == "tool"


def test_parse_taint_rejects_junk():
    assert P.parse_taint("not json") == ("imported",)
    assert P.parse_taint('["web", "bogus"]') == ("web",)
    assert P.from_row("bogus", "x", "[]") == P.IMPORTED


def test_policy_keys_off_taint():
    cfg: dict = {}
    assert P.injectable(P.make("user"), cfg)
    assert P.injectable(P.IMPORTED, cfg)
    assert not P.injectable(None, cfg)  # unknown provenance fails closed
    assert not P.injectable(P.derive(["user", "web"]), cfg)
    assert not P.injectable(P.make("tool"), cfg)
    assert P.rank_multiplier(P.derive(["assistant", "web"]), cfg) == 0.6
    assert P.rank_multiplier(None, cfg) == 1.0
    assert P.may_supersede(P.make("user"), P.make("web"))
    assert not P.may_supersede(P.make("web"), P.make("assistant"))
    assert P.label(P.derive(["user", "web"])) == "web"


# ── schema ───────────────────────────────────────────────────────────────

def _legacy_meta(path: Path) -> sqlite3.Connection:
    """A v0.6 meta db: current schema minus the provenance columns."""
    import common

    conn = sqlite3.connect(path)
    common.ensure_schema(conn)
    for col in ("source", "trust", "taint"):
        conn.execute(f"ALTER TABLE chunks DROP COLUMN {col}")
    conn.execute("INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES ('old1', 'p', 'stop', 1, 'h')")
    conn.commit()
    return conn


def test_lookup_reads_pre_v07_meta_as_imported(tmp_path):
    conn = _legacy_meta(tmp_path / "m.sqlite")
    assert P.lookup(conn, "p", ["old1", "missing"]) == {"old1": P.IMPORTED}


def test_ensure_schema_migrates_legacy_rows_to_imported(tmp_path):
    import common

    conn = _legacy_meta(tmp_path / "m.sqlite")
    common.ensure_schema(conn)
    common.ensure_schema(conn)  # idempotent
    row = conn.execute("SELECT source, trust, taint FROM chunks WHERE id = 'old1'").fetchone()
    assert row == ("imported", 0.5, '["imported"]')
    assert P.lookup(conn, "p", ["old1"]) == {"old1": P.IMPORTED}


def test_all_meta_openers_share_one_schema(tmp_path):
    import dmn
    import dream

    for mod in (dream, dmn):
        pdir = tmp_path / mod.__name__
        pdir.mkdir()
        conn = mod._open_meta_direct(pdir)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        assert {"source", "trust", "taint", "salience", "labile_until"} <= cols
        conn.close()


# ── /remember ────────────────────────────────────────────────────────────

def test_remember_defaults_to_assistant_source(tmp_path, monkeypatch):
    import common
    import remember

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setattr(remember, "_update_backlinks", lambda *a, **k: None)
    a = remember.insert_chunk("model-written note", "note", False, cwd=str(tmp_path))
    u = remember.insert_chunk("user-typed note", "note", False, cwd=str(tmp_path), source="user")
    meta = common.open_meta(str(tmp_path))
    got = P.lookup(meta, common.project_key(str(tmp_path)), [a, u])
    meta.close()
    assert got[a] == P.make("assistant") and got[u] == P.make("user")


def test_remember_cli_joins_unquoted_words(tmp_path, monkeypatch):
    import remember

    seen = {}
    monkeypatch.setattr(remember, "insert_chunk", lambda **kw: seen.update(kw) or "id")
    monkeypatch.setattr(sys, "argv", ["remember.py", "--kind", "note", "--source", "user", "use", "uv", "always"])
    assert remember.main() == 0
    assert seen["text"] == "use uv always" and seen["source"] == "user"
    monkeypatch.setattr(sys, "argv", ["remember.py", "--kind", "note", "plain"])
    remember.main()
    assert seen["source"] == "assistant"


def test_remember_command_is_user_only():
    md = (REPO / "commands" / "remember.md").read_text()
    front = md.split("---")[1]
    assert "disable-model-invocation: true" in front
    assert "remember.py\" --source user $ARGUMENTS" in md


# ── eval carries provenance ──────────────────────────────────────────────

def test_synthetic_corpus_marks_poison_with_untrusted_taint():
    corpus = [json.loads(line) for line in
              (REPO / "tests/fixtures/eval_synthetic/corpus.jsonl").read_text().splitlines() if line]
    poison = [c for c in corpus if c.get("poison")]
    assert len(poison) >= 4
    for c in poison:
        taint = c.get("taint", [c["source"]])
        assert set(taint) & {"web", "tool"}, c["id"]
    qs = [json.loads(line) for line in
          (REPO / "tests/fixtures/eval_synthetic/questions.jsonl").read_text().splitlines() if line]
    named = {pid for q in qs for pid in q.get("poison_ids", [])}
    assert named == {c["id"] for c in poison}


def test_synthetic_build_writes_provenance(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_eval", REPO / "scripts/eval/run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    names = run_eval.build_synthetic(tmp_path, run_eval.Embedder("fake", "", fake_dim=16))
    rows = {}
    for proj in names.values():
        conn = sqlite3.connect(tmp_path / proj / "meta.sqlite")
        rows.update({r[0]: r[1:] for r in conn.execute("SELECT id, source, trust, taint FROM chunks")})
        conn.close()
    assert rows["p01"] == ("web", 0.3, '["web"]')
    assert rows["p03"] == ("distilled", 0.3, '["assistant", "web"]')
    assert rows["a01"] == ("assistant", 0.8, '["assistant"]')
