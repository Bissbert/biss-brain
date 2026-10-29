#!/usr/bin/env python3
"""Validation test for 3.5 log rotation.

Instantiates a RotatingFileHandler pointed at a temp file, logs 50_000 lines,
and asserts that at least one backup file (flush.log.1) was created.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path


def test_rotating_handler_creates_backup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "flush.log"
        backup_path = Path(tmpdir) / "flush.log.1"

        logger = logging.getLogger("test.logrotate")
        logger.setLevel(logging.DEBUG)

        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

        # Log 50_000 lines — each is ~80 chars, total ~4 MB → triggers at least one rotation
        for i in range(50_000):
            logger.info("log line %07d: some test data to fill up the rotating file handler", i)

        handler.close()
        logger.removeHandler(handler)

        assert log_path.exists(), "flush.log was never created"
        assert backup_path.exists(), (
            f"flush.log.1 not created after 50k lines — rotation did not trigger. "
            f"log_path size: {log_path.stat().st_size} bytes"
        )
        print(f"  PASS: flush.log.1 exists ({backup_path.stat().st_size} bytes backup)")
        print(f"  flush.log size: {log_path.stat().st_size} bytes")


def test_flush_py_uses_rotating_handler() -> None:
    """Verify that flush.py's _setup_logging wires a RotatingFileHandler."""
    # Import flush in a way that doesn't trigger actual I/O to ~/.claude/brain
    import importlib, types

    # Patch BRAIN_ROOT before importing flush
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch common.BRAIN_ROOT so _setup_logging writes to tmpdir
        import common
        original_brain_root = common.BRAIN_ROOT
        common.BRAIN_ROOT = Path(tmpdir)

        # Reset cached flush module if already imported
        import importlib
        if "flush" in sys.modules:
            del sys.modules["flush"]

        # Clear root logger handlers to avoid interference
        root = logging.getLogger()
        old_handlers = root.handlers[:]
        root.handlers.clear()

        try:
            import flush  # this calls _setup_logging()
        finally:
            common.BRAIN_ROOT = original_brain_root

        # Check that a RotatingFileHandler is registered on the root logger
        has_rotating = any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in logging.getLogger().handlers
        )
        # Restore handlers
        logging.getLogger().handlers = old_handlers

        assert has_rotating, "flush.py did not register a RotatingFileHandler"
        print("  PASS: flush.py uses RotatingFileHandler")


if __name__ == "__main__":
    print("test_logrotate.py")
    test_rotating_handler_creates_backup()
    test_flush_py_uses_rotating_handler()
    print("\nAll logrotate tests passed.")
