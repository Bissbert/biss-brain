"""v0.7 phase 8.1: the store records its embed fingerprint and refuses a mismatched config."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402

PY = sys.executable


def _cfg(model="nomic-embed-text", dim=768, qp=""):
    return {"embed": {"model": model, "dim": dim, "query_prefix": qp}}


def _write_config(root: Path, model="nomic-embed-text") -> Path:
    f = root / "config.toml"
    f.write_text(f'[distill]\nbackend = "passthrough"\n[embed]\nmodel = "{model}"\ndim = 768\n'
                 '[retrieve]\nhybrid = false\n')
    return f


def _env(root: Path) -> dict:
    return {**os.environ, "BISS_BRAIN_ROOT": str(root), "BISS_BRAIN_CONFIG": str(root / "config.toml"),
            "BISS_BRAIN_FAKE_EMBED": "1", "CLAUDE_PLUGIN_ROOT": str(REPO)}


def test_marker_absent_is_tolerated_and_claimed_only_on_an_empty_store(tmp_path):
    assert common.store_mismatch(_cfg(), tmp_path) is None
    (tmp_path / "abc" / "lancedb").mkdir(parents=True)
    assert not common.claim_store(_cfg(), tmp_path)  # vectors of unknown origin: no fingerprint
    empty = tmp_path / "fresh"
    assert common.claim_store(_cfg(), empty) and not common.claim_store(_cfg(), empty)
    assert common.read_store_marker(empty)["schema_version"] == common.SCHEMA_VERSION


def test_mismatch_names_model_dim_and_prefix(tmp_path):
    common.write_store_marker(_cfg(), tmp_path)
    assert common.store_mismatch(_cfg(), tmp_path) is None
    assert "model" in common.store_mismatch(_cfg(model="bge-m3"), tmp_path)
    assert "dim" in common.store_mismatch(_cfg(dim=1024), tmp_path)
    assert "query_prefix" in common.store_mismatch(_cfg(qp="search_query: "), tmp_path)


def test_schema_only_keeps_no_fingerprint(tmp_path):
    common.write_store_marker(_cfg(), tmp_path, schema_only=True)
    m = common.read_store_marker(tmp_path)
    assert m["schema_version"] == common.SCHEMA_VERSION and "embed" not in m
    assert common.store_mismatch(_cfg(model="anything"), tmp_path) is None


def test_inject_hook_is_silent_on_a_mismatch(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    _write_config(root, model="bge-m3")
    common.write_store_marker(_cfg(), root)
    proj = tmp_path / "proj"
    proj.mkdir()
    hook = json.dumps({"cwd": str(proj), "session_id": "s", "prompt": "how do we deploy the app?"})
    for mode in ("session_start", "user_prompt"):
        r = subprocess.run([PY, str(REPO / "scripts" / "inject_context.py"), mode], input=hook, text=True,
                           env=_env(root), capture_output=True, timeout=60)
        assert r.returncode == 0 and r.stdout == ""


def test_flush_leaves_the_queue_on_a_mismatch(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    _write_config(root, model="bge-m3")
    common.write_store_marker(_cfg(), root)
    q = root / "abc123" / "queue"
    q.mkdir(parents=True)
    ev = q / "0000000001-stop-deadbeef.json"
    ev.write_text("{}")
    r = subprocess.run([PY, str(REPO / "scripts" / "flush.py")], env=_env(root), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and ev.exists()
    assert not (root / "abc123" / "lancedb").exists()


def test_mcp_search_returns_an_error_on_a_mismatch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("brain_mcp_server_fp", REPO / "mcp" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    # other tests reload common; patch the module the server holds
    monkeypatch.setattr(server.common, "BRAIN_ROOT", tmp_path)
    server.common.write_store_marker(_cfg(model="other-model"), tmp_path)
    monkeypatch.setattr(server, "_table", lambda cwd: pytest.fail("store opened"))
    for out in (server.search_memory("deploy"), server.related_to("x")):
        assert "embed settings differ" in out[0]["error"]


def test_a_claimed_store_takes_an_insert_and_a_recall(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    _write_config(root)
    proj = tmp_path / "proj"
    proj.mkdir()
    env = _env(root)
    r = subprocess.run([PY, str(REPO / "scripts" / "remember.py"), "--kind", "decision",
                        "we deploy with rsync to the staging box"], cwd=proj, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert common.read_store_marker(root)["embed"]["model"] == "nomic-embed-text"
    r = subprocess.run([PY, str(REPO / "scripts" / "recall.py"), "deploy rsync staging"], cwd=proj, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "rsync" in r.stdout
    _write_config(root, model="bge-m3")
    r = subprocess.run([PY, str(REPO / "scripts" / "remember.py"), "--kind", "decision", "another"], cwd=proj,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "embed settings differ" in r.stderr
