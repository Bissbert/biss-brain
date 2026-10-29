"""v0.7 Phase 5: UserPromptSubmit injection gate, trivial-prompt skip, budget."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import common  # noqa: E402
import retrieve  # noqa: E402


def _cfg(**r):
    return common._deep_merge(common.DEFAULT_CONFIG, {"retrieve": r})


@pytest.mark.parametrize("text", [
    "", "   ", "/biss-brain:core review", "!git status", "yes do it", "ok", "thanks, looks good",
    "continue please", "go ahead", "lgtm",
])
def test_trivial_prompts_are_skipped(text):
    assert retrieve.trivial_prompt(text, _cfg())


@pytest.mark.parametrize("text", [
    "how does the flush daemon batch events", "why did we pick lancedb", "yes but what about the ollama timeout",
])
def test_topical_prompts_are_not_trivial(text):
    assert not retrieve.trivial_prompt(text, _cfg())


def _row(i, cos=0.0, bm25=0.0, text="x"):
    return {"id": i, "_cosine": cos, "_bm25": bm25, "text": text}


def test_gate_passes_on_cosine_or_bm25():
    rows = [_row("a", cos=0.7), _row("b", bm25=9), _row("c", cos=0.3, bm25=2)]
    got = retrieve.inject_gate(rows, _cfg(inject_min_cosine=0.6, inject_min_bm25=7, inject_top_k=3))
    assert [r["id"] for r in got] == ["a", "b"]


def test_gate_bm25_zero_disables_lexical_side():
    got = retrieve.inject_gate([_row("b", bm25=50)], _cfg(inject_min_cosine=0.6, inject_min_bm25=0, inject_top_k=3))
    assert got == []


def test_gate_without_vector_ignores_cosine():
    rows = [_row("a", cos=0.99), _row("b", bm25=9)]
    got = retrieve.inject_gate(rows, _cfg(inject_min_bm25=7, inject_top_k=3), vector=False)
    assert [r["id"] for r in got] == ["b"]


def test_gate_respects_top_k_and_char_budget():
    rows = [_row(str(i), cos=0.9, text="y" * 400) for i in range(3)]
    assert len(retrieve.inject_gate(rows, _cfg(inject_top_k=1))) == 1
    got = retrieve.inject_gate(rows, _cfg(inject_top_k=3, inject_max_chars=900))
    assert [r["id"] for r in got] == ["0", "1"]
    big = retrieve.inject_gate([_row("z", cos=0.9, text="w" * 5000)], _cfg(inject_max_chars=100))
    assert len(big) == 1 and len(big[0]["text"]) <= 102


def test_defaults_inject_at_most_one_item():
    cfg = common.DEFAULT_CONFIG["retrieve"]
    assert cfg["inject_top_k"] == 1 and cfg["inject_max_k"] <= 3 and cfg["inject_max_chars"] <= 6000


def test_legacy_inject_keys_still_read(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text("[retrieve]\nuser_prompt_inject_min_cosine = 0.42\nuser_prompt_inject_top_k = 2\n")
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(p))
    cfg = common.load_config()
    assert cfg["retrieve"]["inject_min_cosine"] == 0.42 and cfg["retrieve"]["inject_top_k"] == 2


def test_prompt_inject_excludes_ledger_ids(monkeypatch):
    rows = [{"id": i, "kind": "stop", "text": i, "_cosine": 0.9, "_relevance_score": 1.0, "_prov": None}
            for i in ("a", "b")]
    monkeypatch.setattr(retrieve, "retrieve", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setattr(retrieve, "fts_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieve.provenance, "injectable", lambda prov, cfg: True)
    got = retrieve.prompt_inject(None, None, "p", "how do releases work", [0.0], _cfg(), exclude={"a"})
    assert [r["id"] for r in got] == ["b"]


# ── the hook: ledger, kill switch, metrics, deadline, side effects ──────────

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEXT = "LanceDB stores the vectors for every project memory chunk"


def _project(tmp_path, deadline_ms=2500):
    from test_user_prompt_inject import _insert_chunk

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[distill]\nbackend = \"passthrough\"\n[embed]\nmodel = \"fake\"\ndim = 768\n"
        "endpoint = \"http://127.0.0.1:9\"\n"
        f"[retrieve]\nhybrid = false\nuser_prompt_inject = true\ninject_deadline_ms = {deadline_ms}\n")
    cwd = tmp_path / "proj"
    cwd.mkdir()
    cid = _insert_chunk(str(cwd), text=TEXT, vec=common._fake_embed(TEXT, 768))
    return cwd, cfg, cid


def _hook(cwd, cfg, prompt=TEXT, sid="s1"):
    # the in-process root, which other tests may have repointed since conftest set the env
    env = {**os.environ, "BISS_BRAIN_CONFIG": str(cfg), "BISS_BRAIN_FAKE_EMBED": "1",
           "BISS_BRAIN_ROOT": str(common.BRAIN_ROOT)}
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/inject_context.py"), "user_prompt_submit"],
        input=json.dumps({"cwd": str(cwd), "prompt": prompt, "session_id": sid}),
        capture_output=True, text=True, env=env, timeout=30, check=True)
    return out.stdout.strip()


def _metrics(cwd):
    import inject_state

    return inject_state.metrics(common.project_dir(str(cwd)))


def test_hook_injects_once_per_session(tmp_path):
    cwd, cfg, cid = _project(tmp_path)
    first = _hook(cwd, cfg)
    assert TEXT in json.loads(first)["hookSpecificOutput"]["additionalContext"]
    assert _hook(cwd, cfg) == ""  # already in this session's context
    assert _hook(cwd, cfg, sid="s2")  # a new session gets it again
    assert [m["outcome"] for m in _metrics(cwd)] == ["injected", "abstain", "injected"]
    m = _metrics(cwd)[0]
    assert m["vector"] is True and m["n"] == 1 and m["ms"] > 0 and "prompt" not in m


def test_hook_kill_switch_and_trivial_prompts(tmp_path):
    import inject_state

    cwd, cfg, _ = _project(tmp_path)
    assert _hook(cwd, cfg, prompt="yes do it") == ""
    inject_state.set_disabled(common.project_dir(str(cwd)), True)
    assert _hook(cwd, cfg) == ""
    inject_state.set_disabled(common.project_dir(str(cwd)), False)
    assert _hook(cwd, cfg)
    assert [m["outcome"] for m in _metrics(cwd)] == ["trivial", "off", "injected"]


def test_hook_injection_is_not_a_retrieval(tmp_path):
    cwd, cfg, cid = _project(tmp_path)
    meta = common.open_meta(str(cwd))
    meta.execute("UPDATE chunks SET last_accessed = 1 WHERE id = ?", (cid,))
    meta.commit()
    assert _hook(cwd, cfg)
    count, last = meta.execute("SELECT retrieval_count, last_accessed FROM chunks WHERE id = ?", (cid,)).fetchone()
    meta.close()
    assert count == 0 and last > 1


def test_hook_skips_memory_already_in_core(tmp_path, monkeypatch):
    import inject_context

    cwd, cfg, _ = _project(tmp_path)
    monkeypatch.setattr(inject_context, "_core_texts", lambda c: {" ".join(TEXT.lower().split())})
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(cfg))
    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    rows, info = inject_context._select(str(cwd), common.load_config(), TEXT, "s9",
                                        common.project_dir(str(cwd)), 0.0)
    assert rows == [] and info["outcome"] == "abstain"


def test_hook_deadline_prints_nothing(tmp_path, monkeypatch, capsys):
    import inject_context

    cwd, cfg, _ = _project(tmp_path, deadline_ms=100)
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(cfg))

    def slow(*a, **k):
        import time
        time.sleep(1.0)
        return [{"id": "x", "text": "late"}], {"outcome": "injected"}

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(inject_context, "_select", slow)
    monkeypatch.setattr(inject_context.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        inject_context._user_prompt_submit(str(cwd), common.load_config(), {"prompt": TEXT, "session_id": "s1"},
                                           common.project_dir(str(cwd)))
    assert capsys.readouterr().out == ""
    assert _metrics(cwd)[-1]["outcome"] == "deadline"


def test_session_start_resets_ledger_except_on_resume(tmp_path):
    import inject_state

    pdir = tmp_path
    inject_state.ledger_add(pdir, "s1", ["a"])
    inject_state.ledger_add(pdir, "s1", ["b"])
    assert inject_state.ledger_ids(pdir, "s1") == {"a", "b"}
    inject_state.ledger_add(pdir, "s1", ["c"], reset=True)
    assert inject_state.ledger_ids(pdir, "s1") == {"c"}
    inject_state.ledger_add(pdir, "../evil", ["x"])
    assert not list(pdir.rglob("*evil*"))


def test_metrics_summary_percentiles(tmp_path):
    import inject_state

    for ms in range(1, 101):
        inject_state.record(tmp_path, outcome="abstain", ms=float(ms), vector=ms % 10 != 0)
    inject_state.record(tmp_path, outcome="trivial", ms=0.1)
    s = inject_state.summary(tmp_path)
    assert s["n_retrieved"] == 100 and s["p50_ms"] == 50 and s["p95_ms"] == 95
    assert s["embed_fallbacks"] == 10 and s["outcomes"]["trivial"] == 1


def test_doctor_warns_on_slow_prompt_hook(tmp_path, monkeypatch):
    import doctor
    import inject_state

    monkeypatch.chdir(tmp_path)
    cfg = _cfg(user_prompt_inject=True)
    pdir = common.project_dir(str(tmp_path))
    assert doctor.inject_checks(str(tmp_path), cfg) == []
    for _ in range(30):
        inject_state.record(pdir, outcome="abstain", ms=1500.0, vector=False)
    labels = " ".join(w for w, _ in doctor.inject_checks(str(tmp_path), cfg))
    assert "p95 1500 ms" in labels and "embed budget" in labels
    inject_state.set_disabled(pdir, True)
    assert "off for this project" in doctor.inject_checks(str(tmp_path), cfg)[0][0]


def test_inject_command_toggles(tmp_path, monkeypatch, capsys):
    import inject_cmd
    import inject_state

    monkeypatch.chdir(tmp_path)
    inject_cmd.main(["off"])
    assert inject_state.disabled(common.project_dir(str(tmp_path)))
    inject_cmd.main(["on"])
    assert not inject_state.disabled(common.project_dir(str(tmp_path)))
    assert "gate: cosine >= 0.6 or bm25 >= 7.0" in capsys.readouterr().out


@pytest.mark.parametrize("prompt,inject", [("yes do it", "true"), ("how does the daemon batch events", "false")])
def test_cheap_hook_paths_do_not_import_lancedb(tmp_path, prompt, inject):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[retrieve]\nuser_prompt_inject = {inject}\n")
    cwd = tmp_path / "proj"
    (common.project_dir(str(cwd)) / "lancedb").mkdir()  # a warm project, so the cold-store exit is not what saves it
    env = {**os.environ, "BISS_BRAIN_CONFIG": str(cfg), "BISS_BRAIN_ROOT": str(common.BRAIN_ROOT)}
    out = subprocess.run(
        [sys.executable, "-X", "importtime", str(REPO / "scripts/inject_context.py"), "user_prompt_submit"],
        input=json.dumps({"cwd": str(cwd), "prompt": prompt, "session_id": "s1"}),
        capture_output=True, text=True, env=env, timeout=30, check=True)
    assert out.stdout == ""
    assert not [line for line in out.stderr.splitlines() if line.rstrip().endswith(("lancedb", "pyarrow"))]
