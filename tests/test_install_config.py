"""install.py records claude_bin without rewriting the rest of config.toml."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import install  # noqa: E402


def test_example_config_keeps_comments_and_nested_tables(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text((REPO / "config.toml.example").read_text())
    before = cfg.read_text()
    install._toml_set_distill_claude_bin(cfg, "/opt/bin/claude")
    after = cfg.read_text()
    parsed = tomllib.loads(after)
    assert parsed["distill"]["claude_bin"] == "/opt/bin/claude"
    assert parsed["distill"]["models"] == tomllib.loads(before)["distill"]["models"]
    assert after.replace('claude_bin = "/opt/bin/claude"\n', "", 1) == before


def test_updates_an_existing_entry_in_place(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[distill]\n# keep me\nclaude_bin = "/old"\n\n[distill.models]\nstop = "haiku"\n')
    install._toml_set_distill_claude_bin(cfg, '/new "q"')
    assert cfg.read_text() == '[distill]\n# keep me\nclaude_bin = "/new \\"q\\""\n\n[distill.models]\nstop = "haiku"\n'
    assert tomllib.loads(cfg.read_text())["distill"]["claude_bin"] == '/new "q"'


def test_adds_the_section_when_missing(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[embed]\nmodel = "nomic-embed-text"')
    install._toml_set_distill_claude_bin(cfg, "/c")
    parsed = tomllib.loads(cfg.read_text())
    assert parsed == {"embed": {"model": "nomic-embed-text"}, "distill": {"claude_bin": "/c"}}


def test_an_unparseable_file_is_left_alone(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[distill\nbroken")
    with pytest.raises(tomllib.TOMLDecodeError):
        install._toml_set_distill_claude_bin(cfg, "/c")
    assert cfg.read_text() == "[distill\nbroken"
