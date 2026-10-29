#!/usr/bin/env python3
"""Preflight checker for biss-brain.

Verifies all runtime dependencies are in place and prints a pass/fail
summary with remediation hints. Exits 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

# Allow running from scripts/ directly (adds repo root to sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import BRAIN_ROOT, load_config  # noqa: E402

_OK = "✓"
_FAIL = "✗"
_WARN = "!"


def _check(label: str, ok: bool, hint: str = "") -> bool:
    status = _OK if ok else _FAIL
    print(f"  {status}  {label}")
    if not ok and hint:
        print(f"       hint: {hint}")
    return ok


def _warn(label: str, hint: str = "") -> None:
    print(f"  {_WARN}  {label}")
    if hint:
        print(f"       hint: {hint}")


def auto_memory_state(cwd: str, env: dict | None = None, home: Path | None = None) -> tuple[bool, str]:
    """Whether Claude Code's built-in auto memory is on for cwd, and why.

    Documented controls: CLAUDE_CODE_DISABLE_AUTO_MEMORY (1 off, 0 forces on)
    and the autoMemoryEnabled setting (default on in local sessions). Reads
    the user, project and local settings files only; managed settings are
    not checked.
    """
    env = os.environ if env is None else env
    flag = env.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY")
    if flag == "1":
        return False, "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"
    if flag == "0":
        return True, "CLAUDE_CODE_DISABLE_AUTO_MEMORY=0"
    home = home or Path.home()
    try:
        import core

        root = core.repo_root(cwd)
    except Exception:  # noqa: BLE001
        root = Path(cwd)
    state, where = True, "default (on)"
    # later files override earlier ones: user < project < local
    for f in (home / ".claude" / "settings.json", root / ".claude" / "settings.json",
              root / ".claude" / "settings.local.json"):
        try:
            v = json.loads(f.read_text()).get("autoMemoryEnabled")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(v, bool):
            state, where = v, str(f)
    return state, where


def core_checks(cwd: str, cfg: dict) -> list[tuple[str, str]]:
    """(warning, hint) pairs for core memory. Writes nothing beyond relinking a moved repo."""
    import core

    out: list[tuple[str, str]] = []
    ccfg = cfg.get("core", {})
    limit = int(ccfg.get("max_items", 200))
    n = core.count(cwd)
    if n > limit:
        out.append((f"core memory has {n} items; only {limit} are injected",
                    "Trim with /biss-brain:core remove <m:id> or edit the core files."))
    elif n >= limit * float(ccfg.get("warn_ratio", 0.8)):
        out.append((f"core memory at {n}/{limit} items", "Trim before it reaches the limit: /biss-brain:core"))
    shared = core.shared_file(core.repo_root(cwd))
    if shared.exists() and not core.shared_enabled(cwd):
        out.append((f"found {shared} but it is not enabled, so it is not injected",
                    "Read it, then /biss-brain:core shared on to use it."))
    for _, path in core.layers_for(cwd):
        lines = core.unscrubbed(path)
        if lines:
            out.append((f"{path.name} lines {', '.join(map(str, lines[:5]))} look like secrets; "
                        "the core git repo will not auto-commit it", f"Edit {path} to remove them."))
    on, where = auto_memory_state(cwd)
    if on and n:
        out.append(("Claude Code's built-in auto memory is also on: two per-project memories "
                    f"load every session ({where})",
                    'To keep only biss-brain core, set "autoMemoryEnabled": false in ~/.claude/settings.json '
                    "(or run /memory and turn auto memory off)."))
    return out


def inject_checks(cwd: str, cfg: dict) -> list[tuple[str, str]]:
    """(warning, hint) pairs for prompt-time injection, from inject_metrics.jsonl."""
    import inject_state as S
    from common import project_dir

    r = cfg["retrieve"]
    if not r.get("user_prompt_inject", False):
        return []
    pdir = project_dir(cwd)
    if S.disabled(pdir):
        return [("prompt injection is off for this project", "Resume with /biss-brain:inject on.")]
    s = S.summary(pdir)
    out: list[tuple[str, str]] = []
    if s["n_retrieved"] >= 20 and s["p95_ms"] > float(r.get("inject_p95_target_ms", 800)):
        out.append((f"prompt hook p95 {s['p95_ms']:.0f} ms over {s['n_retrieved']} prompts "
                    f"(target {r.get('inject_p95_target_ms', 800)} ms)",
                    "Check Ollama is warm (embed.keep_alive) or turn it off: /biss-brain:inject off"))
    if s["n_retrieved"] >= 20 and s["embed_fallbacks"] > s["n_retrieved"] / 4:
        out.append((f"{s['embed_fallbacks']} of {s['n_retrieved']} prompts missed the {r['inject_embed_ms']} ms "
                    "embed budget and used BM25 only", "Check Ollama: ollama ps"))
    if s["outcomes"].get("deadline"):
        out.append((f"{s['outcomes']['deadline']} prompt(s) hit the {r['inject_deadline_ms']} ms deadline", ""))
    return out


# ── v0.7 checks: each returns (warning, hint) pairs and takes its probes as
# arguments so tests can mock Ollama, launchctl and the claude CLI ──────────

LAUNCHD_PREFIX = "ch.bissbert.biss-brain."


def resolver_checks(cfg: dict, find: Callable[[], Path | None] | None = None) -> list[tuple[str, str]]:
    """The claude CLI the daemons spawn for distillation, found the way they find it."""
    if cfg["distill"].get("backend") != "claude_code":
        return []
    if find is None:
        from cli_resolver import find_claude_cli as find  # noqa: PLC0415
    path = find()
    if path is None:
        return [("the claude CLI is not reachable from the daemons; distillation stores raw passthrough",
                 "Set distill.claude_cli in config.toml or BISS_BRAIN_CLAUDE_CLI to its absolute path.")]
    if not os.access(path, os.X_OK):
        return [(f"{path} is not executable", "Point distill.claude_cli at the launcher (~/.local/bin/claude).")]
    return []


def embed_checks(cfg: dict, root: Path, embed: Callable[[str], list[float]] | None = None,
                 probe: bool = True) -> list[tuple[str, str]]:
    """The model answers with the configured dim, and every table holds vectors of that dim."""
    out: list[tuple[str, str]] = []
    dim = int(cfg["embed"]["dim"])
    embed = embed or (lambda t: common.embed(t, cfg, purpose="query", timeout=15))
    try:
        got = len(embed("biss-brain doctor probe")) if probe else dim
        if got != dim:
            out.append((f"{cfg['embed']['model']} returns {got}-dim vectors, config says {dim}",
                        f"Set embed.dim = {got}, or pick the model the store was built with."))
    except Exception as e:  # noqa: BLE001
        out.append((f"embedding with {cfg['embed']['model']} failed: {type(e).__name__}",
                    "Check Ollama: ollama ps; new memories wait in the backlog until it answers."))
    for ld in sorted(root.glob("*/lancedb")):
        try:
            import lancedb  # noqa: PLC0415

            db = lancedb.connect(str(ld))
            if "chunks" not in db.list_tables().tables:
                continue
            have = db.open_table("chunks").schema.field("vector").type.list_size
        except Exception:  # noqa: BLE001
            continue
        if have != dim:
            out.append((f"{ld.parent.name}: stored vectors are {have}-dim, config says {dim}",
                        common.MISMATCH_HINT))
    return out


def store_checks(cfg: dict, root: Path) -> list[tuple[str, str]]:
    """Schema version, fingerprint presence and FTS coverage. A fingerprint mismatch is main()'s ✗."""
    out: list[tuple[str, str]] = []
    marker = common.read_store_marker(root)
    has_data = any(root.glob("*/meta.sqlite"))
    v = (marker or {}).get("schema_version", 0)
    if marker is None:
        if has_data:
            out.append(("no store.json: a pre-v0.7 store, schema and embed model unverified",
                        "Start fresh (scripts/fresh_start.py) or migrate in place (scripts/run_v07_migrations.sh)."))
    elif v < common.SCHEMA_VERSION:
        out.append((f"store schema {v}, this plugin writes {common.SCHEMA_VERSION}",
                    "Run scripts/run_v07_migrations.sh"))
    elif v > common.SCHEMA_VERSION:
        out.append((f"store schema {v} is newer than this plugin ({common.SCHEMA_VERSION})",
                    "Update the plugin."))
    if marker is not None and "embed" not in marker and has_data:
        out.append(("the store has no embed fingerprint (migrated in place), so a model change goes unnoticed",
                    "A fresh start records it: scripts/fresh_start.py"))
    for ld in sorted(root.glob("*/lancedb")):
        try:
            import lancedb  # noqa: PLC0415

            db = lancedb.connect(str(ld))
            if "chunks" not in db.list_tables().tables:
                continue
            n = common.fts_unindexed_rows(db.open_table("chunks"))
        except Exception:  # noqa: BLE001
            continue
        if n is None:
            out.append((f"{ld.parent.name}: no keyword (FTS) index; search is dense only",
                        "The next flush builds it, or run scripts/run_v07_migrations.sh"))
        elif n > 100:
            out.append((f"{ld.parent.name}: {n} rows outside the FTS index", "The next flush refreshes it."))
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def launchd_checks(plugin_root: Path, agents: Path | None = None,
                   run: Callable[[list[str]], str] = _run) -> list[tuple[str, str]]:
    """Read-only: which jobs are loaded, and whether their plists run this plugin version."""
    agents = agents or Path.home() / "Library" / "LaunchAgents"
    loaded = {ln.split()[-1] for ln in run(["launchctl", "list"]).splitlines() if LAUNCHD_PREFIX in ln}
    out: list[tuple[str, str]] = []
    for plist in sorted(agents.glob(f"{LAUNCHD_PREFIX}*.plist")):
        label = plist.stem
        uid = "$(id -u)"
        reload = (f"launchctl bootout gui/{uid}/{label}; "
                  f"launchctl bootstrap gui/{uid} {plist}")
        try:
            args = json.loads(run(["plutil", "-extract", "ProgramArguments", "json", "-o", "-", str(plist)]))
        except ValueError:
            out.append((f"{label}: cannot read ProgramArguments", f"Check {plist}"))
            continue
        if args and Path(args[0]).parent.name == "bin" and Path(args[0]).name == "run":
            # the install runner picks the live plugin copy itself
            if not Path(args[0]).exists():
                out.append((f"{label}: launchd runner {args[0]} is missing", "Run /biss-brain:install"))
            if label not in loaded:
                out.append((f"{label} is installed but not loaded", f"launchctl bootstrap gui/{uid} {plist}"))
            continue
        scripts = [Path(a) for a in args if a.endswith((".py", ".sh"))]
        if scripts:
            out.append((f"{label} is pinned to one plugin copy and stops after an update",
                        f"Run /biss-brain:install to switch it to the runner, then {reload}"))
        if any(not a.exists() for a in scripts):
            out.append((f"{label} runs a plugin copy that no longer exists",
                        f"Rewrite the plist with /biss-brain:install, then {reload}"))
        elif scripts and not all(str(a).startswith(str(plugin_root) + "/") for a in scripts):
            ver = next((a.parts[i + 1] for a in scripts for i, p in enumerate(a.parts[:-1]) if p == "biss-brain"),
                       "another copy")
            try:
                mine = json.loads((plugin_root / ".claude-plugin" / "plugin.json").read_text())["version"]
            except (OSError, ValueError, KeyError):
                mine = str(plugin_root)
            out.append((f"{label} runs plugin {ver}, not this one ({mine})",
                        f"Rewrite the plist with /biss-brain:install, then {reload}"))
        if label not in loaded:
            out.append((f"{label} is installed but not loaded", f"launchctl bootstrap gui/{uid} {plist}"))
    return out


def dream_checks(root: Path, now: float | None = None, max_age_h: float = 48) -> list[tuple[str, str]]:
    """When the last dream pass finished, and whether an error came after it. Timestamps and levels only."""
    log = root / "dream.log"
    if not log.exists():
        return []
    try:
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 256_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []

    def ts(line: str) -> float | None:
        try:
            return time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            return None

    done = next((ts(ln) for ln in reversed(lines) if "dream pass complete" in ln), None)
    err = next((ts(ln) for ln in reversed(lines) if " ERROR " in ln), None)
    now = time.time() if now is None else now
    out: list[tuple[str, str]] = []
    if done is None:
        out.append(("no completed dream pass in dream.log", "Check the dream job: launchctl list | grep biss-brain"))
    elif now - done > max_age_h * 3600:
        out.append((f"last dream pass finished {(now - done) / 3600:.0f} h ago "
                    f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(done))})",
                    "Check the dream job: launchctl list | grep biss-brain"))
    if err is not None and (done is None or err > done):
        out.append((f"dream.log has an ERROR at {time.strftime('%Y-%m-%d %H:%M', time.localtime(err))} "
                    "after the last completed pass", f"Read {log}"))
    return out


def pending_checks(root: Path) -> list[tuple[str, str]]:
    """Work that is waiting: failed distills, skipped windows, chunks without a vector."""
    import pending  # noqa: PLC0415

    out: list[tuple[str, str]] = []
    if b := pending.backlog(root):
        out.append((f"{len(b)} distill window(s) waiting to retry", "They retry on the next flush."))
    if s := pending.skipped(root):
        out.append((f"{len(s)} window(s) skipped after repeated failures (stored raw)",
                    "See <project>/pending/skipped.jsonl"))
    if m := pending.embed_marks(root):
        out.append((f"{len(m)} project(s) have memories without a vector (keyword search only)",
                    "The next flush embeds them once Ollama answers."))
    n = sum(len(pending.embed_skips(p)) for p in root.iterdir() if p.is_dir()) if root.is_dir() else 0
    if n:
        out.append((f"{n} memories the embedder rejects; they stay keyword-only", ""))
    return out


def daemon_checks(root: Path, plugin_root: Path, root_of: Callable[[int], Path | None] | None = None
                  ) -> list[tuple[str, str]]:
    """A flush daemon running from an older plugin copy than this one."""
    from common import daemon_root, plugin_version  # noqa: PLC0415

    try:
        pid = int((root / "flush.pid").read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return []
    theirs = (root_of or daemon_root)(pid)
    if theirs is None or theirs == plugin_root:
        return []
    old, new = plugin_version(theirs), plugin_version(plugin_root)
    if old is None or new is None or old >= new:
        return []
    return [(f"the flush daemon runs from {'.'.join(map(str, old))}, older than {'.'.join(map(str, new))}",
             "A Claude Code session started before the update keeps starting it. Restart old sessions; "
             "the next capture from an updated session replaces it.")]


def scope_checks(cwd: str, cfg: dict) -> list[tuple[str, str]]:
    """Whether this directory is captured, and whether the configured scope paths exist."""
    import scope  # noqa: PLC0415

    out: list[tuple[str, str]] = []
    sc = cfg.get("scope", {})
    for p in sc.get("include_only", []):
        if not Path(os.path.expanduser(p)).exists():
            out.append((f"scope.include_only path {p} does not exist", "Fix it in config.toml."))
    if not scope.in_scope(cwd, cfg):
        out.append((f"{cwd} is out of scope: nothing is captured or injected here",
                    "See [scope] in config.toml, .claude/brain.toml or a .biss-brain-ignore file."))
    cache = Path(common.BRAIN_ROOT) / scope.CACHE
    want = "".join(f"{t} {p}\n" for t, p in scope.rules(cfg))
    if cache.is_file() and cache.read_text() != want:
        out.append(("scope.cache is stale; the hook launcher uses the old rules until the next hook runs", ""))
    return out


def main() -> int:
    print("biss-brain doctor\n")
    all_ok = True
    cfg = load_config()

    # 1. uv on PATH
    all_ok &= _check(
        "uv on PATH",
        shutil.which("uv") is not None,
        "Install via: brew install uv",
    )

    # 2. .venv/ exists (relative to plugin root)
    plugin_root = Path(__file__).resolve().parent.parent
    venv = plugin_root / ".venv"
    all_ok &= _check(
        ".venv is built",
        (venv / ".biss-ready").is_file(),
        "Run /biss-brain:install (or wait: the hooks build it in the background)",
    )

    # 3. ollama daemon reachable
    ollama_ok = False
    try:
        r = httpx.get(cfg["embed"]["endpoint"] + "/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    all_ok &= _check(
        "Ollama daemon reachable",
        ollama_ok,
        "Start Ollama: ollama serve",
    )

    # 4. configured embed model present
    embed_model_ok = False
    if ollama_ok:
        try:
            r = httpx.get(cfg["embed"]["endpoint"] + "/api/tags", timeout=5)
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            configured = cfg["embed"]["model"].split(":")[0]
            embed_model_ok = configured in models
        except Exception:
            pass
    all_ok &= _check(
        f"Embed model '{cfg['embed']['model']}' present in Ollama",
        embed_model_ok,
        f"Pull it: ollama pull {cfg['embed']['model']}",
    )

    # 5. claude CLI on PATH
    all_ok &= _check(
        "claude CLI on PATH",
        shutil.which("claude") is not None,
        "Install Claude Code CLI or add to PATH. Distillation will fall back to passthrough without it.",
    )

    # 6. ~/.claude/brain/ exists and is writable
    brain_exists = BRAIN_ROOT.exists()
    brain_writable = False
    if brain_exists:
        try:
            testfile = BRAIN_ROOT / ".doctor_write_test"
            testfile.touch()
            testfile.unlink()
            brain_writable = True
        except OSError:
            pass
    else:
        # Try to create it
        try:
            BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
            brain_writable = True
        except OSError:
            pass
    all_ok &= _check(
        f"{BRAIN_ROOT} exists and is writable",
        brain_writable,
        f"Run: mkdir -p {BRAIN_ROOT} && chmod 700 {BRAIN_ROOT}",
    )

    # 7. config.toml exists
    config_path_env = os.environ.get("BISS_BRAIN_CONFIG")
    config_file = Path(config_path_env) if config_path_env else BRAIN_ROOT / "config.toml"
    config_exists = config_file.exists()
    all_ok &= _check(
        f"config.toml exists at {config_file}",
        config_exists,
        f"Copy example: cp {plugin_root}/config.toml.example {config_file}",
    )

    # 8. the store's embed fingerprint matches the config (a mismatch fails the run)
    mismatch = common.store_mismatch(cfg)
    all_ok &= _check("store embed settings match config", mismatch is None,
                     f"{mismatch}. Fix: {common.MISMATCH_HINT}")

    # 9. everything else is a warning and never fails the run
    cwd = os.getcwd()
    sections: tuple[tuple[str, Callable[[], list[tuple[str, str]]]], ...] = (
        ("store", lambda: store_checks(cfg, BRAIN_ROOT)),
        ("embedding", lambda: embed_checks(cfg, BRAIN_ROOT, probe=ollama_ok and embed_model_ok)),
        ("distill CLI", lambda: resolver_checks(cfg)),
        ("launchd jobs", lambda: launchd_checks(plugin_root) if sys.platform == "darwin" else []),
        ("dream", lambda: dream_checks(BRAIN_ROOT)),
        ("pending work", lambda: pending_checks(BRAIN_ROOT)),
        ("flush daemon", lambda: daemon_checks(BRAIN_ROOT, plugin_root)),
        ("scope", lambda: scope_checks(cwd, cfg)),
        ("core memory", lambda: core_checks(cwd, cfg)),
        ("prompt injection", lambda: inject_checks(cwd, cfg)),
    )
    for title, check in sections:
        try:
            warnings = check()
        except Exception as e:  # noqa: BLE001
            warnings = [(f"{title} check skipped: {type(e).__name__}", "")]
        if warnings:
            print()
            print(f"  {title}:")
            for label, hint in warnings:
                _warn(label, hint)

    print()
    if all_ok:
        print(f"  {_OK}  All checks passed. biss-brain is ready.")
    else:
        print(f"  {_FAIL}  Some checks failed. Fix the issues above and re-run.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
