"""Subprocess driver: run one production read path over the fixture.

  python _retrieval_driver.py <recall|mcp|inject> <root>

Environment comes from _retrieval_fixture.env_for(). The fixture is rebuilt
before every query, so side effects (RIF, retrieval counts) from one query
never leak into the next. Prints a JSON map {query: [ids in output order]}.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _retrieval_fixture as fx  # noqa: E402

sys.path.insert(0, str(fx.SCRIPTS))


def _fresh(root: Path) -> Path:
    from common import project_key

    cwd = root / "work"
    cwd.mkdir(exist_ok=True)
    shutil.rmtree(root / project_key(cwd), ignore_errors=True)
    fx.build(cwd)
    return cwd


def run_recall(root: Path) -> dict:
    import recall

    out = {}
    for q in fx.QUERIES:
        cwd = _fresh(root)
        os.chdir(cwd)
        sys.argv = ["recall.py", q, "--top-k", "4"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            recall.main()
        out[q] = re.findall(r"id=([0-9a-f]{8})", buf.getvalue())
    return out


def run_mcp(root: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location("brain_mcp_server", fx.SCRIPTS.parent / "mcp" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    server._embed = fx.fake_vec
    out = {}
    for q in fx.QUERIES:
        cwd = _fresh(root)
        os.chdir(cwd)
        out[q] = [r["id"][:8] for r in server.search_memory(q, top_k=4)]
    return out


def run_inject(root: Path) -> dict:
    import inject_context

    out = {}
    for q in fx.QUERIES:
        cwd = _fresh(root)
        sys.argv = ["inject_context.py", "user_prompt_submit"]
        sys.stdin = io.StringIO(json.dumps({"cwd": str(cwd), "prompt": q}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inject_context.main()
        txt = buf.getvalue()
        ctx = json.loads(txt)["hookSpecificOutput"]["additionalContext"] if txt.strip() else ""
        out[q] = re.findall(r"variant (\d+)", ctx)
    return out


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2])
    res = {"recall": run_recall, "mcp": run_mcp, "inject": run_inject}[mode](root)
    sys.stdout = sys.__stdout__
    print("@@JSON@@" + json.dumps(res))
