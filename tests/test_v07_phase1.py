"""v0.7 phase 1: filter safety, cosine gates, FTS coverage, project scoping, scrubbing,
prompt rendering, the single CLI path and judge verdicts."""
from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import lance_filter as F  # noqa: E402
from scrub import scrub  # noqa: E402

DIM = 8


def _unit(*xs: float) -> list[float]:
    import common

    return common.normalize(list(xs) + [0.0] * (DIM - len(xs)))


def _row(cid: str, proj: str, vec: list[float], text: str = "", kind: str = "note", ts: int = 1) -> dict:
    return {
        "id": cid, "project": proj, "session_id": "", "agent_id": "", "kind": kind, "ts": ts,
        "text": text or f"text of {cid}", "content_hash": cid, "superseded_by": "", "vector": vec,
    }


# ── lance_filter ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["x' OR '1'='1", "a b", "", "a;b", "é", "a" * 65, None, 5])
def test_filter_rejects_unsafe_ids(bad):
    with pytest.raises(F.FilterError):
        F.eq("id", bad)


def test_filter_rejects_unknown_column_kind_and_non_int_ts():
    with pytest.raises(F.FilterError):
        F.eq("text", "x")
    with pytest.raises(F.FilterError):
        F.eq("kind", "note' OR '1'='1")
    with pytest.raises(F.FilterError):
        F.ge("ts", "1")
    with pytest.raises(F.FilterError):
        F.ge("ts", True)
    with pytest.raises(F.FilterError):
        F.isin("id", [])


def test_filter_builds_expected_sql():
    assert F.live_in("abc123", F.eq("kind", "note"), None) == (
        "project = 'abc123' AND superseded_by = '' AND kind = 'note'"
    )
    assert F.isin("id", ["a", "b"]) == "id IN ('a', 'b')"
    assert F.eq("superseded_by", "") == "superseded_by = ''"


# ── cosine metric on unit vectors ────────────────────────────────────────

def test_embed_returns_unit_vectors(monkeypatch):
    import common

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    v = common.embed("hello", {"embed": {"dim": 64}})
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_vector_search_reports_cosine_and_gate_fires(tmp_path):
    import common

    tbl = common.open_or_create_chunks(tmp_path / "lancedb", DIM)
    tbl.add([
        _row("same", "p", _unit(1, 0)),
        _row("near", "p", _unit(1, 0.2)),
        _row("orth", "p", _unit(0, 1)),
    ])
    rows = {r["id"]: r["_cosine"] for r in common.vector_search(tbl, _unit(1, 0), F.live_in("p"), 3)}
    assert rows["same"] == pytest.approx(1.0, abs=1e-4)
    assert rows["near"] == pytest.approx(1 / (1.04 ** 0.5), abs=1e-4)
    assert rows["orth"] == pytest.approx(0.0, abs=1e-4)
    # the 0.70 backlink gate keeps the first two only (the old L2 < 0.78 gate never fired
    # on Ollama's norm-15 vectors)
    from backlinks import SIMILARITY_THRESHOLD

    assert {k for k, c in rows.items() if c >= SIMILARITY_THRESHOLD} == {"same", "near"}


# ── FTS coverage ─────────────────────────────────────────────────────────

def test_refresh_fts_covers_rows_added_later(tmp_path):
    import common

    tbl = common.open_or_create_chunks(tmp_path / "lancedb", DIM)
    tbl.add([_row("a", "p", _unit(1, 0), "postgres sixteen upgrade")])
    common.refresh_fts(tbl)
    tbl.add([_row("b", "p", _unit(0, 1), "kubernetes ingress controller")])
    common.refresh_fts(tbl)
    assert common.fts_unindexed_rows(tbl) == 0
    hits = tbl.search("kubernetes", query_type="fts").limit(5).to_list()
    assert [h["id"] for h in hits] == ["b"]


# ── project scoping (S2) ─────────────────────────────────────────────────

def _two_project_meta(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chunks (id TEXT, project TEXT, related_ids TEXT DEFAULT '', "
        "cross_links TEXT DEFAULT '[]', pinned INTEGER DEFAULT 0)"
    )
    conn.executemany("INSERT INTO chunks (id, project) VALUES (?, ?)", [("c1", "projA"), ("c1", "projB")])
    conn.commit()
    return conn


def test_typed_edge_write_is_project_scoped(tmp_path):
    from backlinks import _add_typed_edge, _add_typed_edge_multi

    conn = _two_project_meta(tmp_path / "m.sqlite")
    _add_typed_edge(conn, "c1", "x", "co_occurs", 1.0, 5, project="projA")
    _add_typed_edge_multi(conn, "c1", ["y"], "co_occurs", 1.0, 5, project="projA")
    got = dict(conn.execute("SELECT project, related_ids FROM chunks").fetchall())
    assert "x" in got["projA"] and "y" in got["projA"]
    assert got["projB"] == ""


def test_cross_link_write_is_project_scoped(tmp_path):
    from recall import _write_cross_link

    conn = _two_project_meta(tmp_path / "m.sqlite")
    _write_cross_link(conn, "c1", "g1", 0.9, ts=5, project="projA")
    got = dict(conn.execute("SELECT project, cross_links FROM chunks").fetchall())
    assert "g1" in got["projA"]
    assert got["projB"] == "[]"


def _load_mcp_server():
    spec = importlib.util.spec_from_file_location("biss_brain_mcp_server", REPO / "mcp" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_project(cwd: Path, cid: str) -> None:
    import common

    meta = common.open_meta(str(cwd))
    meta.execute(
        "INSERT INTO chunks (id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, last_accessed) "
        "VALUES (?, ?, '', '', 'note', 1, ?, '', 1)",
        (cid, common.project_key(str(cwd)), cid),
    )
    meta.commit()
    meta.close()


def test_mcp_pin_rejects_chunk_of_another_project(tmp_path, monkeypatch):
    import common

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _seed_project(a, "chunkA")
    server = _load_mcp_server()

    monkeypatch.chdir(b)
    assert server.pin("chunkA") == {"id": "chunkA", "error": "no such chunk in this project"}
    monkeypatch.chdir(a)
    assert server.pin("chunkA") == {"id": "chunkA", "pinned": True}
    meta = common.open_meta(str(a))
    assert meta.execute("SELECT pinned FROM chunks WHERE id = 'chunkA'").fetchone()[0] == 1
    meta.close()


def test_forget_ignores_chunk_of_another_project(tmp_path, monkeypatch, capsys):
    import common
    import forget

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _seed_project(a, "abcdef01")
    monkeypatch.chdir(b)
    monkeypatch.setattr(sys, "argv", ["forget.py", "abcdef", "x' OR 1=1 --"])
    assert forget.main() == 0
    out = capsys.readouterr().out
    assert "no match: abcdef" in out and "not an id" in out
    meta = common.open_meta(str(a))
    assert meta.execute("SELECT superseded_by FROM chunks WHERE id = 'abcdef01'").fetchone()[0] == ""
    meta.close()


def test_post_retrieval_effects_stay_in_project(tmp_path):
    import common
    from effects import apply_post_retrieval

    tbl = common.open_or_create_chunks(tmp_path / "lancedb", DIM)
    tbl.add([_row("c1", "projA", _unit(1, 0)), _row("c2", "projA", _unit(1, 0.1))])
    conn = sqlite3.connect(tmp_path / "m.sqlite")
    conn.execute(
        "CREATE TABLE chunks (id TEXT, project TEXT, retrieval_count INTEGER DEFAULT 0, "
        "retrieval_count_local INTEGER DEFAULT 0, last_accessed INTEGER DEFAULT 0, "
        "labile_until INTEGER DEFAULT 0, decay_score REAL DEFAULT 1.0, pinned INTEGER DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE edge_bumps (project TEXT, chunk_a TEXT, chunk_b TEXT, bumped_at INTEGER, "
        "session_id TEXT, drained INTEGER DEFAULT 0)"
    )
    conn.executemany("INSERT INTO chunks (id, project) VALUES (?, ?)",
                     [("c1", "projA"), ("c2", "projA"), ("c1", "projB"), ("c2", "projB")])
    cfg = dict(common.DEFAULT_CONFIG)
    rows = [dict(r) for r in common.fetch_by_ids(tbl, ["c1", "c2"])]
    apply_post_retrieval(conn, tbl, "projA", rows, cfg, "s1")
    counts = conn.execute(
        "SELECT project, SUM(retrieval_count), SUM(retrieval_count_local), MAX(labile_until) "
        "FROM chunks GROUP BY project ORDER BY project"
    ).fetchall()
    assert counts[0][:3] == ("projA", 2, 2) and counts[0][3] > 0
    assert counts[1] == ("projB", 0, 0, 0)
    assert conn.execute("SELECT DISTINCT project FROM edge_bumps").fetchall() == [("projA",)]


# ── scrubbing (W8, W9) ───────────────────────────────────────────────────

@pytest.mark.parametrize("secret, label", [
    ("sk-ant-api03-" + "Ab3_dE-" * 6, "API_KEY"),
    ("sk-proj-" + "Zx9_" * 8, "API_KEY"),
    ("github_pat_" + "A1b2C3d4E5" * 4, "GH_TOKEN"),
    ("gho_" + "a1B2c3D4e5" * 3 + "a1B2c3", "GH_TOKEN"),
    ("glpat-" + "aB3dE5fG7h" * 2, "GITLAB_TOKEN"),
])
def test_scrub_new_token_patterns(secret, label):
    out = scrub(f"before {secret} after", entropy_check=False)
    assert secret not in out and f"[REDACTED:{label}]" in out
    assert out.startswith("before ") and out.endswith(" after")


def test_scrub_pem_block_and_url_credentials():
    pem = "-----BEGIN RSA " "PRIVATE KEY-----\nMIIEow\nIBAAK\n-----END RSA PRIVATE KEY-----"
    out = scrub(f"key:\n{pem}\ndone", entropy_check=False)
    assert "MIIEow" not in out and "[REDACTED:PRIVATE_KEY]" in out and out.endswith("done")
    assert "MIIE" not in scrub("-----BEGIN " "PRIVATE KEY-----\nMIIEtruncated", entropy_check=False)
    url = scrub("git clone https://bob:hunter2@git.example.com/r.git", entropy_check=False)
    assert "hunter2" not in url and url.endswith("@git.example.com/r.git")
    assert scrub("see https://example.com/a:b", entropy_check=False) == "see https://example.com/a:b"


def test_remember_scrubs_before_storing(tmp_path, monkeypatch):
    import common
    import remember

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setattr(remember, "_update_backlinks", lambda *a, **k: None)
    secret = "glpat-" + "aB3dE5fG7h" * 2
    cid = remember.insert_chunk(f"deploy token is {secret}", "note", False, cwd=str(tmp_path))
    dim = common.load_config()["embed"]["dim"]
    row = common.fetch_by_ids(remember._open_table(str(tmp_path), dim), [cid])[0]
    assert secret not in row["text"] and "[REDACTED:GITLAB_TOKEN]" in row["text"]
    assert common.fts_unindexed_rows(remember._open_table(str(tmp_path), dim)) == 0


# ── UI search filters ────────────────────────────────────────────────────

def test_ui_lance_where_has_only_lance_columns():
    sys.path.insert(0, str(REPO))
    from ui.routers.search import _build_lance_where, _meta_filter

    w = _build_lance_where(False, "note", 10, 20, "sess-1")
    assert w == "superseded_by = '' AND kind = 'note' AND session_id = 'sess-1' AND ts >= 10 AND ts <= 20"
    assert _build_lance_where(True, "", None, None, None) is None
    assert _build_lance_where(True, "bogus' OR 1=1", None, None, "bad session") is None
    assert _meta_filter({"pinned": 1, "salience": 50}, True, 40, 0)
    assert not _meta_filter({"pinned": 0}, True, 0, 0)
    assert not _meta_filter({"pinned": 1, "salience": 10}, True, 40, 0)
    assert not _meta_filter({"consolidation_round": 1}, False, 0, 2)



# ── prompt rendering (W1) ────────────────────────────────────────────────

LITERAL = "code {zz_literal} and {new_text} braces"


def test_fill_is_single_pass_and_strict():
    import distill

    out = distill.fill("A={old_text} B={new_text}", old_text=LITERAL, new_text="n")
    assert out == f"A={LITERAL} B=n"  # the {new_text} inside old_text is not substituted
    with pytest.raises(KeyError):
        distill.fill("A={old_text} B={new_text}", old_text="o")
    assert distill.fill('{"verdict": "X"} {text}', text="t") == '{"verdict": "X"} t'


@pytest.mark.parametrize("kind", sorted(__import__("distill").PROMPT_FILES))
def test_every_prompt_template_renders_completely(kind):
    import distill

    template = distill._load_prompt(kind)
    keys = distill.placeholders(template)
    assert keys, f"{kind} template has no placeholders"
    out = distill.render(kind, **{k: f"<{k}:{{zz_literal}}>" for k in keys})
    assert not (distill.placeholders(out) - {"zz_literal"}), kind
    for k in keys:
        assert f"<{k}:{{zz_literal}}>" in out


@pytest.fixture
def sent(monkeypatch):
    """Capture every prompt that reaches `claude -p`; reply with *reply* (default "note")."""
    import subprocess
    from types import SimpleNamespace

    import cli_resolver
    import distill

    prompts: list[dict] = []
    state = {"reply": json_reply("note")}

    def fake_run(cmd, **kw):
        prompts.append({"prompt": cmd[cmd.index("-p") + 1], "cmd": cmd, "cwd": kw.get("cwd"),
                        "stdin": kw.get("stdin")})
        return SimpleNamespace(returncode=0, stdout=state["reply"], stderr="")

    monkeypatch.setattr(cli_resolver, "find_claude_cli", lambda: Path("/fake/claude"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(distill, "load_config", lambda: {
        "distill": {"backend": "claude_code", "models": {}, "ollama": {"model": "x"}},
        "embed": {"endpoint": "http://127.0.0.1:9"},
    })
    monkeypatch.delenv("BISS_BRAIN_FAKE_DISTILL", raising=False)
    return SimpleNamespace(prompts=prompts, state=state)


def json_reply(result: str, **extra) -> str:
    import json

    return json.dumps({"result": result, **extra})


def _assert_clean(prompt: str, kind: str) -> None:
    import distill

    leftover = distill.placeholders(prompt) & distill.placeholders(distill._load_prompt(kind))
    assert not leftover, f"{kind}: {leftover} reached the LLM"
    assert "{zz_literal}" in prompt  # caller-supplied braces survive verbatim


def test_salience_prompt_reaches_llm_rendered(sent):
    import flush

    flush._recent_texts_cache["texts"] = ["earlier {zz_literal} chunk"]
    sent.state["reply"] = json_reply('{"score": 7, "reason": "r"}')
    cfg = {"capture": {"salience": {"use_llm": True}}}
    assert flush.compute_salience("new {zz_literal} chunk", _unit(1), [], cfg) == 7
    assert len(sent.prompts) == 1
    _assert_clean(sent.prompts[0]["prompt"], "salience")
    assert "new {zz_literal} chunk" in sent.prompts[0]["prompt"]


def test_merge_stop_judge_arc_classify_prompts_reach_llm_rendered(sent, monkeypatch):
    import distill
    import dream
    import judge
    import remember

    distill.ask("merge", old_text="old {zz_literal}", new_text="new {zz_literal}")
    distill.distill("stop", "session {zz_literal}")
    judge.judge_conflict("new {zz_literal}", "old {zz_literal}")
    dream._distill_arc("arc {zz_literal}", "haiku", kind="arc")
    dream._distill_arc("day {zz_literal}", "haiku", kind="daily")
    remember._classify_kind("todo {zz_literal}")
    kinds = ["merge", "stop", "judge", "arc", "daily", "classify"]
    assert len(sent.prompts) == len(kinds)
    for kind, p in zip(kinds, sent.prompts):
        _assert_clean(p["prompt"], kind)


# ── one CLI path (W2/W3) ─────────────────────────────────────────────────

def test_run_claude_is_bare_toolless_and_runs_in_brain_root(sent):
    import cli
    import common

    assert cli.run_claude("hi", "haiku") == "note"
    c = sent.prompts[0]
    assert c["cmd"][0] == "/fake/claude"
    cmd = c["cmd"]
    # --bare would ignore a keychain login; isolation comes from these instead
    assert "--bare" not in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == "" and cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd and cmd.index("--tools") > cmd.index("-p") + 1
    assert c["cmd"][c["cmd"].index("--model") + 1] == cli.MODEL_MAP["haiku"]
    assert c["cwd"] == str(common.BRAIN_ROOT)
    assert c["stdin"] == subprocess.DEVNULL


def test_run_claude_is_error_reply_is_a_failure(sent):
    import cli

    before = cli._FAILURE_COUNTER["claude_cli_error"]
    sent.state["reply"] = json_reply("Not logged in", is_error=True)
    assert cli.run_claude("hi") is None
    assert cli._FAILURE_COUNTER["claude_cli_error"] == before + 1


def test_no_bare_claude_subprocess_left_in_scripts():
    offenders = [
        f"{p.name}:{i}"
        for p in (REPO / "scripts").rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if '"claude",' in line or "'claude'," in line
    ]
    assert offenders == []


# ── judge verdicts (W6) ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw, verdict", [
    ('{"verdict": "contradicts", "action": "x", "reason": "y"}', "CONTRADICTS"),
    ('```json\n{"Verdict": "Extends", "reason": "adds"}\n```', "EXTENDS"),
    ("UNRELATED\n", "UNRELATED"),
    ('{"verdict": "DEFER"}', "DEFER"),
    ('{"verdict": "maybe"}', None),
    ("I think they are related", None),
    ("", None),
])
def test_parse_verdict(raw, verdict):
    import judge

    assert judge.parse_verdict(raw)[0] == verdict


def test_judge_reply_in_requested_json_shape_supersedes(sent):
    import judge

    sent.state["reply"] = json_reply('{"verdict": "contradicts", "action": "supersede", "reason": "r"}')
    assert judge.judge_conflict("uses JWT", "uses sessions") == "CONTRADICTS"


@pytest.mark.parametrize("reply", [json_reply("no idea"), json_reply("x", is_error=True)])
def test_judge_failure_is_none_never_a_verdict(sent, reply):
    import judge

    sent.state["reply"] = reply
    assert judge.judge_conflict("a", "b") is None
    out = judge.judge_with_project_context(
        {"text": "a", "project": "p1"}, {"text": "b", "project": "p2"}, False, {}
    )
    assert out["verdict"] is None


def test_judge_cross_project_never_contradicts_or_extends(sent):
    import judge

    for word, expected in (("CONTRADICTS", "DIVERGES"), ("EXTENDS", "UNRELATED")):
        sent.state["reply"] = json_reply(f'{{"verdict": "{word}"}}')
        out = judge.judge_with_project_context(
            {"text": "a", "project": "p1"}, {"text": "b", "project": "p2"}, False, {}
        )
        assert out["verdict"] == expected
    assert "{same_project}" not in sent.prompts[-1]["prompt"]
    assert "SAME_PROJECT: false" in sent.prompts[-1]["prompt"]


# ── remember scrubs before the classifier sees the text (W8) ─────────────

def test_remember_main_scrubs_before_classify(monkeypatch):
    import remember

    secret = "glpat-" + "aB3dE5fG7h" * 2
    seen: dict[str, str] = {}
    monkeypatch.setattr(remember, "_classify_kind", lambda t: seen.setdefault("classify", t) and "note")
    monkeypatch.setattr(remember, "insert_chunk", lambda text, kind, pinned, **k: seen.setdefault("insert", text) and "id")
    monkeypatch.setattr(sys, "argv", ["remember", f"token {secret}"])
    assert remember.main() == 0
    assert secret not in seen["classify"] and secret not in seen["insert"]

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
