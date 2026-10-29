#!/usr/bin/env python3
"""/biss-brain:core — show, review and curate core memory.

  core [show]                 the core items for this project, by layer
  core review                 open proposals (from the model and the dream pass)
  core accept <id> [--layer private|global]
  core reject <id>
  core remove <m:id>          retire a core item (same as deleting its line)
  core shared on|off          opt this repo's .claude/brain/CORE.md in or out
  core export                 the CLAUDE.md @import line for this repo's core file

Only this user-typed command and /remember --core write live core.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import core  # noqa: E402
from scrub import scrub  # noqa: E402


def _tilde(p: Path) -> str:
    home = Path.home()
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def cmd_show(cwd: str, cfg: dict) -> int:
    core.sync(cwd)
    layers = core.layers_for(cwd)
    if not layers:
        print("No core memory for this project yet. Add one with /biss-brain:remember --core <text>.")
    total = 0
    for layer, path in layers:
        items = core.load(path).items()
        total += len(items)
        print(f"## {layer} — {_tilde(path)} ({len(items)})")
        for it in items:
            print(f"- [{it.section}] {scrub(it.text)}  (m:{it.id}, {it.source}, {it.date})")
        print()
    shared = core.shared_file(core.repo_root(cwd))
    if shared.exists() and not core.shared_enabled(cwd):
        print(f"Note: {_tilde(shared)} exists but is not enabled; it is not injected. "
              "Enable with /biss-brain:core shared on after reading it.")
    limit = int(cfg.get("core", {}).get("max_items", 200))
    print(f"{total}/{limit} items injected at session start.")
    n = len(core.proposals(core.proposal_projects(cwd)))
    if n:
        print(f"{n} open proposal(s): /biss-brain:core review")
    return 0


def cmd_review(cwd: str) -> int:
    rows = core.proposals(core.proposal_projects(cwd))
    if not rows:
        print("No open core proposals.")
        return 0
    for r in rows:
        print(f"- {r['id']}  {core.describe(r)}")
    path = core.render_proposed(cwd)
    print()
    print("Accept: /biss-brain:core accept <id>   Reject: /biss-brain:core reject <id>")
    if path:
        print(f"(also written to {_tilde(path)})")
    return 0


def cmd_export(cwd: str) -> int:
    p = core.private_file(cwd)
    if p is None:
        print("No private core file for this repo yet.", file=sys.stderr)
        return 1
    print(f"@{_tilde(p)}")
    print()
    print("Paste that line into a CLAUDE.md to load this core file without the plugin.")
    print("While biss-brain is enabled it already injects core at session start, so the")
    print("import would load it twice. From a project CLAUDE.md the import is outside the")
    print("project, so Claude Code asks once to approve it; from ~/.claude/CLAUDE.md it does not.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="/biss-brain:core", description="Curate core memory.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("show")
    sub.add_parser("review")
    a = sub.add_parser("accept")
    a.add_argument("id")
    a.add_argument("--layer", choices=["private", "global"], default=None)
    sub.add_parser("reject").add_argument("id")
    sub.add_parser("remove").add_argument("id")
    sub.add_parser("shared").add_argument("state", choices=["on", "off"])
    sub.add_parser("export")
    args = p.parse_args(argv)
    cwd = os.getcwd()
    cfg = common.load_config()
    try:
        if args.cmd in (None, "show"):
            return cmd_show(cwd, cfg)
        if args.cmd == "review":
            return cmd_review(cwd)
        if args.cmd == "accept":
            print(core.accept(cwd, args.id, args.layer))
            core.render_proposed(cwd)
            return 0
        if args.cmd == "reject":
            core.reject(cwd, args.id)
            core.render_proposed(cwd)
            print(f"rejected {args.id}")
            return 0
        if args.cmd == "remove":
            mid = args.id.removeprefix("m:")
            path = core.remove(cwd, mid)
            print(f"removed m:{mid} from {_tilde(path)}" if path else f"no core item m:{mid}")
            return 0 if path else 1
        if args.cmd == "shared":
            core.enable_shared(cwd, args.state == "on")
            print(f"shared layer {args.state} for {core.repo_root(cwd).name}")
            return 0
        if args.cmd == "export":
            return cmd_export(cwd)
    except core.CoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
