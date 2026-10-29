#!/usr/bin/env python3
"""Launcher for the biss-brain UI inspector.

Usage:
  ui.py            # start (or reuse running) server and open browser
  ui.py --stop     # terminate the running server
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate plugin root from CLAUDE_PLUGIN_ROOT env or relative to this file.
# ---------------------------------------------------------------------------
_PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT", "")).resolve()
if not _PLUGIN_ROOT.is_dir():
    _PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# Add scripts/ to path so we can import common helpers
_SCRIPTS_DIR = _PLUGIN_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from common import BRAIN_ROOT, project_key  # noqa: E402

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
_PID_FILE = BRAIN_ROOT / "ui.pid"
_PORT_FILE = BRAIN_ROOT / "ui.port"
_LOG_FILE = BRAIN_ROOT / "ui.log"


def _read_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _read_port() -> int | None:
    try:
        return int(_PORT_FILE.read_text().strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _port_reachable(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_healthz(port: int, timeout: float = 5.0) -> bool:
    """Poll /healthz until it returns 200 or timeout elapses."""
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _start_server() -> int:
    """Spawn uvicorn in the background, return the port it's listening on."""
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    port = _free_port()

    argv = [
        sys.executable, "-m", "uvicorn",
        "ui.server:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "warning",
    ]
    if os.environ.get("BISS_BRAIN_DEV"):
        argv += [
            "--reload",
            "--reload-dir", str(_PLUGIN_ROOT / "ui"),
            "--reload-dir", str(_PLUGIN_ROOT / "scripts"),
        ]

    with open(_LOG_FILE, "ab") as logf:
        proc = subprocess.Popen(
            argv,
            cwd=str(_PLUGIN_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
        )

    _PID_FILE.write_text(str(proc.pid))
    _PORT_FILE.write_text(str(port))
    return port


def _cmd_stop() -> None:
    pid = _read_pid()
    if pid is None:
        print("biss-brain UI: no pidfile found, server may not be running.")
        return

    if not _pid_alive(pid):
        print(f"biss-brain UI: process {pid} is not running, cleaning up.")
        _PID_FILE.unlink(missing_ok=True)
        _PORT_FILE.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    else:
        # Force-kill
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _PID_FILE.unlink(missing_ok=True)
    _PORT_FILE.unlink(missing_ok=True)
    print(f"biss-brain UI: stopped (pid {pid}).")


def main() -> None:
    if "--stop" in sys.argv:
        _cmd_stop()
        return

    project_hash = project_key(os.getcwd())
    url_suffix = f"/?project={project_hash}"

    # Check if an existing server is still alive
    pid = _read_pid()
    port = _read_port()
    if pid is not None and port is not None and _pid_alive(pid) and _port_reachable(port):
        url = f"http://127.0.0.1:{port}{url_suffix}"
        print(url)
        webbrowser.open(url)
        return

    # Start fresh
    print("biss-brain UI: starting server…")
    port = _start_server()

    if not _wait_for_healthz(port, timeout=5.0):
        print(
            f"biss-brain UI: server did not become healthy within 5s. "
            f"Check {_LOG_FILE} for errors.",
            file=sys.stderr,
        )
        sys.exit(1)

    url = f"http://127.0.0.1:{port}{url_suffix}"
    print(url)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
