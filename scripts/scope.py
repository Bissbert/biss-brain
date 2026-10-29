"""Which directories the hooks capture and inject in.

A directory is out of scope when:
  - it matches `[scope] exclude` ("~" and "/" match exactly, every other entry
    matches the directory and everything below it),
  - `[scope] include_only` is set and the directory is under none of its entries,
  - it or a parent holds `.biss-brain-ignore`, or a `.claude/brain.toml` with
    `enabled = false`.

`_run.sh` applies the same rules in shell from `<brain>/scope.cache`, so an
excluded directory exits before Python starts. Python writes the cache
(`refresh_cache`) whenever a hook gets this far, and checks again itself, so
a missing or stale cache only costs one Python start.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import common

CACHE = "scope.cache"
IGNORE_FILE = ".biss-brain-ignore"
_EXACT = {"~", "/"}


def _real(p: str) -> str:
    return os.path.realpath(os.path.expanduser(p))


def rules(cfg: dict) -> list[tuple[str, str]]:
    """[(type, realpath)]: e = exclude exactly, x = exclude the tree, i = include only."""
    sc = cfg.get("scope", {})
    out = [("e", "/")]
    for e in sc.get("exclude", []):
        out.append(("e" if e.strip() in _EXACT else "x", _real(e)))
    out += [("i", _real(i)) for i in sc.get("include_only", [])]
    return out


def _under(d: str, root: str) -> bool:
    return d == root or d.startswith(root.rstrip("/") + "/")


def _opted_out(d: str) -> bool:
    p = Path(d)
    for q in (p, *p.parents):
        if (q / IGNORE_FILE).exists():
            return True
        t = q / ".claude" / "brain.toml"
        if t.is_file():
            try:
                if tomllib.loads(t.read_text()).get("enabled", True) is False:
                    return True
            except (OSError, ValueError):
                pass
    return False


def in_scope(cwd: str, cfg: dict) -> bool:
    d = _real(cwd)
    inc = []
    for t, p in rules(cfg):
        if (t == "e" and d == p) or (t == "x" and _under(d, p)):
            return False
        if t == "i":
            inc.append(p)
    if inc and not any(_under(d, p) for p in inc):
        return False
    return not _opted_out(d)


def refresh_cache(cfg: dict) -> None:
    """Write `<brain>/scope.cache` for `_run.sh` when its rules changed (never raises)."""
    text = "".join(f"{t} {p}\n" for t, p in rules(cfg))
    f = Path(common.BRAIN_ROOT) / CACHE
    try:
        if f.is_file() and f.read_text() == text:
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(f)
    except OSError:
        pass


def check(cwd: str, cfg: dict) -> bool:
    """Hook entry: refresh the shell cache, then answer in_scope."""
    refresh_cache(cfg)
    return in_scope(cwd, cfg)
