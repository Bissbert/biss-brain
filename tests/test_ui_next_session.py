"""v0.9 overview: the next-session preview is what the SessionStart hook
injects, and building it writes nothing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import httpx

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "scripts", REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")

import common  # noqa: E402

TEXT = "The flush daemon batches events every five seconds"


def _project(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str]:
    from test_user_prompt_inject import _insert_chunk

    cfg = tmp_path / "config.toml"
    cfg.write_text("[embed]\nmodel = \"fake\"\ndim = 768\nendpoint = \"http://127.0.0.1:9\"\n")
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(cfg))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _insert_chunk(str(cwd), text=TEXT, vec=common._fake_embed(TEXT, 768))
    proj = common.project_key(str(cwd))
    (common.BRAIN_ROOT / proj / "project_path.txt").write_text(str(cwd))
    import core

    core.add(str(cwd), "Ship every step as its own commit")
    return cwd, cfg, proj


def _hook(cwd: Path, cfg: Path) -> str:
    env = {**os.environ, "BISS_BRAIN_CONFIG": str(cfg), "BISS_BRAIN_FAKE_EMBED": "1",
           "BISS_BRAIN_ROOT": str(common.BRAIN_ROOT)}
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/inject_context.py"), "session_start"],
        input=json.dumps({"cwd": str(cwd), "session_id": "s1", "source": "startup"}),
        capture_output=True, text=True, env=env, timeout=60, check=True)
    return json.loads(out.stdout)["hookSpecificOutput"]["additionalContext"]


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    # SQLite creates its -shm/-wal sidecars for any reader of a WAL database
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")
            if p.is_file() and not p.name.endswith(("-shm", "-wal"))}


def test_preview_equals_what_the_hook_injects(tmp_path, monkeypatch):
    cwd, cfg, proj = _project(tmp_path, monkeypatch)
    from ui.preview import next_session

    before = _snapshot(common.BRAIN_ROOT)
    got = next_session(proj)
    assert _snapshot(common.BRAIN_ROOT) == before  # looking changes nothing
    assert got["status"] == "ok" and got["core_shown"] == 1 and len(got["memories"]) == 1
    assert got["text"] == _hook(cwd, cfg)


def test_overview_shows_the_next_session(tmp_path, monkeypatch):
    _, _, proj = _project(tmp_path, monkeypatch)
    from ui.server import app

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.get(f"/?project={proj}")

    page = " ".join(anyio.run(run).text.split())
    assert "Next session" in page and "1 core note and 1 recent memory" in page
    assert TEXT in page and "Exact text the session-start hook injects" in page
    assert "&#127760;" not in page and "border-left:3px" not in page


def test_a_label_that_hashes_elsewhere_is_ignored(tmp_path, monkeypatch):
    _, _, proj = _project(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    (common.BRAIN_ROOT / proj / "project_path.txt").write_text(str(other))
    from ui.preview import next_session

    got = next_session(proj)
    assert got["cwd"] is None and got["core_shown"] == 0 and len(got["memories"]) == 1
