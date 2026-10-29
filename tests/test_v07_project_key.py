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
