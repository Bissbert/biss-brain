"""Tests for worktree-aware project_key in scripts/common.py.

Covers:
- A fake worktree layout where the .git file points back to the main repo;
  project_key(worktree_dir) must equal project_key(main_repo).
- A regular project path (no .claude/worktrees/ segment) hashes identically
  before and after the normalization logic (idempotent / non-worktree paths
  are untouched).
- A detached HEAD worktree still resolves to the main repo (the .git file
  content is the same regardless of HEAD state).
- A path whose .git file contains a non-existent gitdir falls back gracefully
  and returns the original path's hash.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _key(p: Path) -> str:
    """Recompute the project_key the same way common.py does, for assertions."""
    return hashlib.sha256(str(p).encode()).hexdigest()[:12]


def _setup_fake_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal fake worktree layout inside tmp_path.

    Layout::

        tmp_path/
          main/                          ← canonical repo root
            .git/
              worktrees/
                wt1/                     ← what git creates for the linked worktree
          main/.claude/
            worktrees/
              wt1/                       ← the actual worktree checkout directory
                .git                     ← text file: "gitdir: <main>/.git/worktrees/wt1"

    Returns (worktree_dir, main_repo_root).
    """
    main = tmp_path / "main"
    # Create the main repo's .git/worktrees/wt1 directory
    gitdir_entry = main / ".git" / "worktrees" / "wt1"
    gitdir_entry.mkdir(parents=True)

    # Create the worktree checkout directory inside .claude/worktrees/wt1
    worktree_dir = main / ".claude" / "worktrees" / "wt1"
    worktree_dir.mkdir(parents=True)

    # Write the .git file that links the worktree back to the main repo
    git_file = worktree_dir / ".git"
    git_file.write_text(f"gitdir: {gitdir_entry}\n", encoding="utf-8")

    return worktree_dir, main


# ---------------------------------------------------------------------------
# Helpers — reload common.py fresh so module-level state doesn't bleed across
# ---------------------------------------------------------------------------

def _import_fresh_common():
    """Import common with a clean module state."""
    if "common" in sys.modules:
        del sys.modules["common"]
    import common  # noqa: PLC0415
    return common


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_worktree_key_equals_main_repo_key(tmp_path: Path) -> None:
    """project_key(worktree_dir) must equal project_key(main_repo_root)."""
    worktree_dir, main_repo = _setup_fake_worktree(tmp_path)

    common = _import_fresh_common()

    key_wt = common.project_key(str(worktree_dir))
    key_main = common.project_key(str(main_repo))

    assert key_wt == key_main, (
        f"Worktree key {key_wt!r} differs from main repo key {key_main!r}. "
        f"worktree_dir={worktree_dir}, main_repo={main_repo}"
    )


def test_non_worktree_path_is_stable(tmp_path: Path) -> None:
    """A regular project path (no .claude/worktrees/ segment) is unaffected."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    common = _import_fresh_common()

    key = common.project_key(str(project_dir))
    # The expected key is sha256 of the resolved path — no normalization applied
    expected = _key(project_dir.resolve())

    assert key == expected, (
        f"Non-worktree path was unexpectedly remapped: got {key!r}, "
        f"expected {expected!r}"
    )


def test_worktree_with_detached_head(tmp_path: Path) -> None:
    """Detached HEAD worktrees use the same .git file format — should still resolve."""
    worktree_dir, main_repo = _setup_fake_worktree(tmp_path)
    # In a detached HEAD the .git file content is the same; HEAD is just a
    # different ref. Our parsing is content-agnostic so this test reuses the
    # same layout and confirms equality holds.
    common = _import_fresh_common()

    key_wt = common.project_key(str(worktree_dir))
    key_main = common.project_key(str(main_repo))

    assert key_wt == key_main


def test_worktree_missing_gitdir_falls_back(tmp_path: Path) -> None:
    """If the gitdir path in .git does not exist, fall back to the worktree path."""
    worktree_dir = tmp_path / "main" / ".claude" / "worktrees" / "wt2"
    worktree_dir.mkdir(parents=True)

    # Point .git at a non-existent location
    git_file = worktree_dir / ".git"
    git_file.write_text("gitdir: /nonexistent/path/.git/worktrees/wt2\n", encoding="utf-8")

    common = _import_fresh_common()

    key = common.project_key(str(worktree_dir))
    # Should fall back to hashing the worktree_dir itself (resolved)
    expected = _key(worktree_dir.resolve())

    assert key == expected, (
        f"Expected fallback key {expected!r}, got {key!r}"
    )


def test_normalize_worktree_helper_directly(tmp_path: Path) -> None:
    """Unit-test _normalize_worktree directly for the main happy path."""
    worktree_dir, main_repo = _setup_fake_worktree(tmp_path)

    common = _import_fresh_common()

    normalized = common._normalize_worktree(worktree_dir.resolve())
    assert normalized == main_repo.resolve(), (
        f"Expected {main_repo.resolve()}, got {normalized}"
    )


def test_normalize_worktree_no_op_for_plain_path(tmp_path: Path) -> None:
    """_normalize_worktree returns the input unchanged for a plain directory."""
    plain = tmp_path / "some" / "project"
    plain.mkdir(parents=True)

    common = _import_fresh_common()

    result = common._normalize_worktree(plain)
    assert result == plain
