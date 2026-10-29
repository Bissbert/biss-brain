"""Test isolation: redirect BRAIN_ROOT to a per-session tmp dir.

The production `common.BRAIN_ROOT` defaults to `~/.claude/brain` — running tests
against that path leaks fixture data into the user's real brain store. This
hook sets BISS_BRAIN_ROOT to a fresh tmp dir BEFORE any test module is
imported, so common.py's module-level evaluation reads the override.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest


def pytest_configure(config):
    """Runs after conftest is loaded, before test collection imports test modules."""
    tmp = tempfile.mkdtemp(prefix="biss-brain-tests-")
    os.environ["BISS_BRAIN_ROOT"] = tmp
    config._biss_brain_tmp = tmp


def pytest_unconfigure(config):
    """Runs after the entire session — wipe the tmp brain root."""
    tmp = getattr(config, "_biss_brain_tmp", None)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


_PATCHED_MODULE_NAMES = ("common", "ui.data", "scripts.common")
_BRAIN_ROOT_BASELINE: dict[str, object] = {}


def _capture_baselines() -> None:
    """Eagerly import the modules and remember their pristine BRAIN_ROOT values."""
    if _BRAIN_ROOT_BASELINE:
        return
    for name in _PATCHED_MODULE_NAMES:
        try:
            __import__(name)
        except Exception:
            continue
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "BRAIN_ROOT"):
            _BRAIN_ROOT_BASELINE[name] = mod.BRAIN_ROOT


@pytest.fixture(autouse=True)
def _restore_brain_root():
    """Restore module state between tests.

    Tests mutate `common.BRAIN_ROOT` (and re-export on `ui.data`), monkey-
    patch `ui.data.validate_project` and similar attributes, and replace
    router-level functions. Without isolation the next test inherits stale
    paths and mock callables, breaking everything that resolves projects.

    Strategy: capture pristine BRAIN_ROOT once. After each test, restore
    BRAIN_ROOT and evict every `ui.*` module so the next test re-imports
    them clean. `common` stays cached for speed; only its BRAIN_ROOT is
    reset to baseline.
    """
    _capture_baselines()
    yield
    # Restore BRAIN_ROOT on any still-loaded shared module
    for name, value in _BRAIN_ROOT_BASELINE.items():
        mod = sys.modules.get(name)
        if mod is not None:
            setattr(mod, "BRAIN_ROOT", value)
    # Evict ui.* — tests freely monkey-patch routers/data; cheapest reset
    # is to drop the modules and let the next test re-import.
    for name in list(sys.modules.keys()):
        if name == "ui" or name.startswith("ui."):
            del sys.modules[name]
