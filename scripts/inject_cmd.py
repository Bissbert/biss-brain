#!/usr/bin/env python3
"""/biss-brain:inject — prompt-time injection for this project.

  inject [status]   on/off, gate settings, latency and outcomes from recent prompts
  inject off        stop injecting memory into prompts in this project
  inject on         resume (still subject to retrieve.user_prompt_inject)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import inject_state as S  # noqa: E402


def status(pdir: Path, cfg: dict) -> int:
    r = cfg["retrieve"]
    on = r.get("user_prompt_inject", False)
    print(f"prompt injection: {'on' if on else 'off'} (retrieve.user_prompt_inject)")
    if S.disabled(pdir):
        print("this project: off (/biss-brain:inject on to resume)")
    print(f"gate: cosine >= {r['inject_min_cosine']} or bm25 >= {r['inject_min_bm25']}, "
          f"top {r['inject_top_k']}, {r['inject_max_chars']} chars, "
          f"embed {r['inject_embed_ms']} ms, deadline {r['inject_deadline_ms']} ms")
    s = S.summary(pdir)
    if not s["n"]:
        print("no prompts recorded yet")
        return 0
    lat = (f"p50 {s['p50_ms']:.0f} ms, p95 {s['p95_ms']:.0f} ms over {s['n_retrieved']} retrievals"
           if s["n_retrieved"] else "no retrievals yet")
    print(f"last {s['n']} prompts: {lat}; embed fallbacks {s['embed_fallbacks']}")
    print("outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="/biss-brain:inject", description="Prompt-time memory injection.")
    p.add_argument("state", nargs="?", choices=["status", "on", "off"], default="status")
    a = p.parse_args(argv)
    cfg = common.load_config()
    pdir = common.project_dir(os.getcwd())
    if a.state != "status":
        S.set_disabled(pdir, a.state == "off")
        print(f"prompt injection {a.state} for this project")
    return status(pdir, cfg)


if __name__ == "__main__":
    sys.exit(main())
