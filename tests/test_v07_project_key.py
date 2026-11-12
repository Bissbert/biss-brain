"""v0.7 phase 6.3: one project key per repo, read from .git without spawning git."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402


def _git(*args, cwd):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-c", "protocol.file.allow=always", *args], cwd=cwd, env=env, check=True,
                   capture_output=True, timeout=30)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "main"
    r.mkdir()
    _git("init", "-q", "-b", "main", cwd=r)
    (r / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=r)
    _git("commit", "-q", "-m", "init", cwd=r)
    return r.resolve()


def test_subdirectories_and_any_worktree_share_the_main_key(repo, tmp_path):
    (repo / "src" / "deep").mkdir(parents=True)
    _git("worktree", "add", "-q", str(repo / ".trees" / "feat"), "-b", "feat", cwd=repo)
    _git("worktree", "add", "-q", str(tmp_path / "elsewhere"), "-b", "other", cwd=repo)
    (tmp_path / "elsewhere" / "pkg").mkdir()
    key = common.project_key(repo)
    for d in (repo / "src" / "deep", repo / ".trees" / "feat", tmp_path / "elsewhere" / "pkg"):
        assert common.repo_root(d) == repo
        assert common.project_key(d) == key


def test_relative_gitdir_and_symlinked_cwd(repo, tmp_path):
    rel = ["--relative-paths"] if _supports_relative(repo) else []
    _git("worktree", "add", "-q", *rel, str(repo / ".trees" / "rel"), cwd=repo)
    link = tmp_path / "link"
    link.symlink_to(repo / ".trees" / "rel")
    assert common.repo_root(link) == repo


def _supports_relative(repo):
    r = subprocess.run(["git", "worktree", "add", "-h"], cwd=repo, capture_output=True, text=True)
    return "--relative-paths" in r.stdout + r.stderr


def test_a_submodule_is_its_own_project(repo, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _git("init", "-q", "-b", "main", cwd=lib)
    (lib / "l.txt").write_text("l")
    _git("add", "l.txt", cwd=lib)
    _git("commit", "-q", "-m", "lib", cwd=lib)
    _git("submodule", "add", "-q", str(lib), "vendor/lib", cwd=repo)
    sub = repo / "vendor" / "lib"
    assert (sub / ".git").is_file()
    assert common.repo_root(sub) == sub and common.project_key(sub) != common.project_key(repo)
    assert common.repo_root(repo / "vendor") == repo


def test_no_repo_and_bare_repo_fall_back_to_realpath(tmp_path):
    plain = tmp_path / "plain" / "x"
    plain.mkdir(parents=True)
    assert common.repo_root(plain) == plain.resolve()
    bare = tmp_path / "bare.git"
    _git("init", "-q", "--bare", str(bare), cwd=tmp_path)
    assert common.repo_root(bare) == bare.resolve()


def test_home_is_never_a_project_root(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)  # a dotfiles repo in $HOME
    (home / "proj").mkdir()
    monkeypatch.setenv("HOME", str(home))
    assert common.repo_root(home / "proj") == (home / "proj").resolve()


def test_matches_git_for_this_checkout():
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO / "scripts",
                         capture_output=True, text=True, check=True).stdout.strip()
    assert common.repo_root(REPO / "scripts") == Path(out).resolve().parent


# ── outside git: the highest marked ancestor is the root ─────────────────

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(tmp_path / "absent.toml"))
    return h.resolve()


def _mk(p, *markers):
    p.mkdir(parents=True, exist_ok=True)
    for m in markers:
        (p / m).mkdir() if m == ".claude" else (p / m).write_text("x")
    return p


def test_subfolders_of_a_marked_project_share_its_key(home):
    proj = _mk(home / "minecraft-modpack", "README.md")
    deep = _mk(proj / "pack" / "kubejs" / "server_scripts")
    assert common.repo_root(deep) == proj
    assert common.project_key(deep) == common.project_key(proj)


def test_the_highest_marker_wins_over_a_nearer_one(home):
    top = _mk(home / "x" / "courses", ".claude")
    sub = _mk(top / "bash", "package.json", "README.md")
    assert common.repo_root(_mk(sub / "content" / "lessons")) == top


def test_a_folder_of_projects_without_markers_attracts_nothing(home):
    workspace = _mk(home / "sources" / "projects" / "experiments")
    lone = _mk(workspace / "lone" / "notes")
    assert common.repo_root(workspace) == workspace
    assert common.repo_root(lone) == lone  # no marker anywhere: keeps its realpath


def test_claude_md_alone_is_not_a_project_marker(home):
    ws = _mk(home / "plugins", "CLAUDE.md")
    assert common.repo_root(_mk(ws / "scratch")) == ws / "scratch"


def test_generic_roots_and_home_are_never_crossed(home):
    _mk(home / "Documents", "README.md")  # a marked generic root
    _mk(home, "README.md")
    job = _mk(home / "Documents" / "Karriere", "README.md")
    assert common.repo_root(_mk(job / "03" / "_tracking")) == job
    assert common.repo_root(_mk(home / "Documents" / "loose")) == home / "Documents" / "loose"


def test_configured_generic_roots_stop_the_walk(home, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f'[scope]\ngeneric_roots = ["{home}/a/ws"]\n')
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(cfg))
    _mk(home / "a", "README.md")
    ws = _mk(home / "a" / "ws")
    assert common.repo_root(_mk(ws / "p")) == ws / "p"


def test_root_marker_wins_outright(home):
    top = _mk(home / "big", "README.md")
    mid = _mk(top / "apps" / "one", common.ROOT_MARKER)
    assert common.repo_root(_mk(mid / "src")) == mid


def test_a_git_repo_under_a_marked_folder_is_still_its_own_project(home):
    top = _mk(home / "big", "README.md")
    r = _mk(top / "repo")
    _git("init", "-q", "-b", "main", cwd=r)
    assert common.repo_root(_mk(r / "src")) == r
