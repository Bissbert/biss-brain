#!/usr/bin/env python3
"""Full biss-brain installer.

Runs the same steps as bootstrap.sh plus optional DMN daemon install,
schema migration trigger, and a doctor-style summary at the end.

Designed to run with the system Python: uses only the stdlib for the
prerequisite phase, then invokes the freshly-created .venv/bin/python
for the migration step.

Usage:
    python3 install.py            # install dream pass + core
    python3 install.py --with-dmn # also install DMN idle daemon
    python3 install.py --reinstall  # force re-copy of plists / config
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT") or Path(__file__).resolve().parent.parent).resolve()
BRAIN_ROOT = Path.home() / ".claude" / "brain"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
# The launchd jobs run this stable copy of scripts/launchd_run.sh, which finds
# the live plugin copy, so a plugin update does not strand them on an old path.
RUNNER = BRAIN_ROOT / "bin" / "run"

_OK = "✓"
_FAIL = "✗"
_INFO = "→"


def info(msg: str) -> None:
    print(f"  {_INFO}  {msg}")


def ok(msg: str) -> None:
    print(f"  {_OK}  {msg}")


def fail(msg: str, hint: str = "") -> None:
    print(f"  {_FAIL}  {msg}")
    if hint:
        print(f"       hint: {hint}")


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, **kw)


def step_uv_sync() -> bool:
    if not have("uv"):
        fail("uv on PATH", "Install via: brew install uv")
        return False
    info("building the venv with uv sync (this may take a few minutes)...")
    # build_venv.sh holds the per-copy lock, so a build the hooks already
    # started in the background is waited for, not run twice.
    proc = run(["bash", str(PLUGIN_ROOT / "scripts" / "build_venv.sh"), "--wait"],
               env={**os.environ, "CLAUDE_PLUGIN_ROOT": str(PLUGIN_ROOT)})
    if proc.returncode != 0:
        fail(f"uv sync (exit {proc.returncode})")
        return False
    ok("dependencies installed (.venv ready)")
    return True


def step_ollama() -> bool:
    if not have("ollama"):
        fail("ollama on PATH", "Install via: brew install ollama")
        return False
    # Check daemon reachable via curl (avoids httpx until venv is ready)
    if have("curl"):
        proc = run(["curl", "-sf", "http://127.0.0.1:11434/api/tags"], capture_output=True)
        if proc.returncode != 0:
            fail("ollama daemon reachable", "Start it: ollama serve &")
            return False
    ok("ollama daemon reachable")
    info("pulling nomic-embed-text (768-dim) — skipped if already present")
    proc = run(["ollama", "pull", "nomic-embed-text"])
    if proc.returncode != 0:
        fail(f"ollama pull nomic-embed-text (exit {proc.returncode})")
        return False
    ok("nomic-embed-text available")
    return True


def step_brain_dir(reinstall: bool) -> bool:
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = BRAIN_ROOT / "config.toml"
    example = PLUGIN_ROOT / "config.toml.example"
    if reinstall and cfg.exists():
        cfg.unlink()
    if not cfg.exists():
        if not example.exists():
            fail(f"config.toml.example missing at {example}")
            return False
        shutil.copy(example, cfg)
        ok(f"installed default config at {cfg}")
    else:
        ok(f"config exists at {cfg}")
    return True


DREAM_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ch.bissbert.biss-brain.dream</string>

    <key>ProgramArguments</key>
    <array>
        <string>{runner}</string>
        <string>dream.py</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_path}</string>

    <key>StandardErrorPath</key>
    <string>{log_path}</string>

    <key>KeepAlive</key>
    <false/>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""

DMN_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ch.bissbert.biss-brain.dmn</string>

    <key>ProgramArguments</key>
    <array>
        <string>{runner}</string>
        <string>dmn.py</string>
    </array>

    <key>KeepAlive</key>
    <true/>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{log_path}</string>

    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""


def _install_plist(label: str, contents: str, reinstall: bool) -> bool:
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    dst = LAUNCH_AGENTS / f"{label}.plist"
    if dst.exists() and not reinstall:
        run(["launchctl", "unload", str(dst)], capture_output=True)
    dst.write_text(contents)
    proc = run(["launchctl", "load", "-w", str(dst)], capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"launchctl load {label}", proc.stderr.strip() or "see launchctl error")
        return False
    ok(f"loaded {label}")
    return True


def step_runner() -> bool:
    RUNNER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PLUGIN_ROOT / "scripts" / "launchd_run.sh", RUNNER)
    RUNNER.chmod(0o755)
    (BRAIN_ROOT / "active_root").write_text(f"{PLUGIN_ROOT}\n")
    ok(f"launchd runner at {RUNNER} (follows plugin updates)")
    return True


def step_dream_plist(reinstall: bool) -> bool:
    contents = DREAM_PLIST_TEMPLATE.format(
        runner=str(RUNNER),
        log_path=str(BRAIN_ROOT / "dream.log"),
    )
    return _install_plist("ch.bissbert.biss-brain.dream", contents, reinstall)


def step_dmn_plist(reinstall: bool) -> bool:
    contents = DMN_PLIST_TEMPLATE.format(
        runner=str(RUNNER),
        log_path=str(BRAIN_ROOT / "dmn.log"),
    )
    return _install_plist("ch.bissbert.biss-brain.dmn", contents, reinstall)


def step_legacy_cleanup() -> bool:
    legacy = LAUNCH_AGENTS / "ch.bissbert.biss-brain.decay.plist"
    if legacy.exists():
        run(["launchctl", "unload", str(legacy)], capture_output=True)
        legacy.unlink()
        ok("removed legacy decay plist (decay is now Stage 1c of dream)")
    return True


def step_migrate() -> bool:
    """Trigger schema migration by opening _global meta with the venv'd python."""
    venv_python = PLUGIN_ROOT / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        fail(f"venv python missing at {venv_python}", "uv sync must run first")
        return False
    code = (
        "import sys; sys.path.insert(0, 'scripts');"
        " from common import open_meta, BRAIN_ROOT;"
        " BRAIN_ROOT.mkdir(parents=True, exist_ok=True);"
        " (BRAIN_ROOT / '_global').mkdir(parents=True, exist_ok=True);"
        " open_meta(str(BRAIN_ROOT / '_global'));"
        " print('migration ok')"
    )
    proc = run(
        [str(venv_python), "-c", code],
        cwd=str(PLUGIN_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail("schema migration", proc.stderr.strip()[:200] or "see error above")
        return False
    ok("schema migration applied to _global")
    return True


def step_claude_cli() -> bool:
    if not have("claude"):
        fail(
            "claude CLI on PATH",
            "Distillation will fall back to passthrough without it. Install: https://claude.ai/code",
        )
        return False
    ok("claude CLI on PATH")
    return True


def _toml_set_distill_claude_bin(cfg_path: Path, bin_path: str) -> None:
    """Write (or update) ``[distill] claude_bin`` in *cfg_path*.

    Edits the one line in place, so comments, nested tables such as
    [distill.models] and every other key stay as they are. The result is
    parsed before it is written; a file that would not parse is left alone.
    """
    import tomllib

    escaped = bin_path.replace("\\", "\\\\").replace('"', '\\"')
    entry = f'claude_bin = "{escaped}"\n'
    lines = cfg_path.read_text().splitlines(keepends=True) if cfg_path.exists() else []
    header = next((i for i, ln in enumerate(lines) if ln.strip() == "[distill]"), None)
    if header is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n", "[distill]\n", entry]
    else:
        end = next((i for i in range(header + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        found = next((i for i in range(header + 1, end)
                      if lines[i].split("=", 1)[0].strip() == "claude_bin"), None)
        if found is None:
            lines.insert(header + 1, entry)
        else:
            lines[found] = entry
    text = "".join(lines)
    tomllib.loads(text)  # raises before anything is written
    cfg_path.write_text(text)


def step_record_claude_bin() -> bool:
    """Resolve the claude CLI and persist its path into config.toml.

    This ensures that future LaunchAgent runs (which inherit a stripped PATH)
    can still find the CLI via the config-based probe in cli_resolver.
    Returns True always — a missing CLI is a warning, not an install failure.
    """
    # Import from the venv so we get the real module, not system Python's view.
    # Fall back to a direct import if the venv python isn't available yet.
    venv_python = PLUGIN_ROOT / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        info("venv not ready; skipping claude_bin recording")
        return True

    code = (
        "import sys; sys.path.insert(0, 'scripts');"
        " from cli_resolver import find_claude_cli;"
        " r = find_claude_cli();"
        " print(str(r) if r else '')"
    )
    proc = run(
        [str(venv_python), "-c", code],
        cwd=str(PLUGIN_ROOT),
        capture_output=True,
        text=True,
    )
    resolved = proc.stdout.strip() if proc.returncode == 0 else ""

    cfg_path = BRAIN_ROOT / "config.toml"
    if resolved:
        try:
            _toml_set_distill_claude_bin(cfg_path, resolved)
            ok(f"recorded claude_bin = {resolved} in config.toml")
        except Exception as exc:  # noqa: BLE001
            fail(f"could not write claude_bin to config.toml: {exc}")
    else:
        print(
            f"\n  {_FAIL}  WARNING: claude CLI not found during install.\n"
            f"       Distillation will be disabled until you either:\n"
            f"         1. Install Claude Code CLI: https://claude.ai/code\n"
            f"         2. Set [distill] claude_bin = \"/path/to/claude\" in\n"
            f"            {cfg_path}\n"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Install biss-brain")
    parser.add_argument("--with-dmn", action="store_true", help="also install DMN idle daemon")
    parser.add_argument("--reinstall", action="store_true", help="force re-copy of plists / config")
    args = parser.parse_args()

    print(f"biss-brain installer  (plugin root: {PLUGIN_ROOT})\n")

    if sys.platform != "darwin":
        info("non-macOS detected — launchd plists will be skipped")

    all_ok = True
    all_ok &= step_uv_sync()
    all_ok &= step_ollama()
    all_ok &= step_brain_dir(args.reinstall)
    all_ok &= step_claude_cli()  # warning only, doesn't fail install

    if sys.platform == "darwin":
        step_legacy_cleanup()
        all_ok &= step_runner()
        all_ok &= step_dream_plist(args.reinstall)
        if args.with_dmn:
            all_ok &= step_dmn_plist(args.reinstall)
        else:
            info("DMN daemon NOT installed (use --with-dmn to enable idle maintenance)")

    if (PLUGIN_ROOT / ".venv" / "bin" / "python").is_file():
        all_ok &= step_migrate()

    # Record resolved claude CLI path so LaunchAgent contexts can find it.
    step_record_claude_bin()

    print()
    if all_ok:
        print(f"  {_OK}  biss-brain installed.  config: {BRAIN_ROOT}/config.toml")
        print(f"       run /biss-brain:doctor to re-verify, /biss-brain:ui to inspect")
    else:
        print(f"  {_FAIL}  install completed with issues — fix the items above and re-run /biss-brain:install")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
