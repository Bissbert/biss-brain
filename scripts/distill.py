"""Distillation backends and prompt rendering.

Public API:
  - render(kind, **vars) -> str        fill a prompts/ template in one pass
  - call(prompt, model) -> str | None  send an already-rendered prompt
  - ask(kind, model=None, **vars)      render + call
  - distill(kind, text, model)         session-excerpt distillation (stop, subagent, pre_compact)

Backends:
  - claude_code: `claude -p` headless via cli.run_claude. Uses the user's subscription.
  - ollama:      local Ollama model (qwen2.5-coder default).
  - passthrough: distill() returns truncated raw text; call() returns None. For debugging.

`[distill] fallback` lists backends distill() tries, in order, when the
primary one returns nothing (claude_code or ollama; passthrough is refused, it
would store raw transcript text). Only distill() falls back: the conflict
judge and dream use the primary backend.

The failure counter lives in cli.py and is re-exported here for the status pill.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from cli import _FAILURE_COUNTER, _record_failure, run_claude  # noqa: F401 — re-exported
from common import load_config

log = logging.getLogger("biss-brain.distill")

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILES = {
    "stop": "distill_stop.md",
    "subagent_stop": "distill_subagent.md",
    "pre_compact": "distill_precompact.md",
    "salience": "score_salience.md",
    "merge": "reconsolidate_merge.md",
    "arc": "dream_arc.md",
    "daily": "dream_daily.md",
    "classify": "classify_kind.md",
    "judge": "judge_conflict.md",
}

_FALLBACK_PROMPT = (
    "Summarize the following Claude Code session excerpt for long-term memory.\n"
    "Extract: decisions made, files touched, errors hit, dead ends, open questions.\n"
    "Output 3-8 bullet points. Be concrete; cite file paths and identifiers.\n\n---\n{text}\n"
)

# Placeholders are lowercase identifiers in braces. JSON examples in the
# templates ({"verdict": ...}) never match because of the quote.
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

# Truncation constants
_TRUNC_MAX = 8000
_TRUNC_HEAD = 4000
_TRUNC_TAIL = 3500


def _smart_truncate(text: str) -> str:
    """Truncate *text* to at most _TRUNC_MAX bytes, preserving message boundaries.

    If ``len(text) <= _TRUNC_MAX`` the original string is returned unchanged.
    Otherwise the first _TRUNC_HEAD bytes and last _TRUNC_TAIL bytes are kept,
    separated by a human-readable marker that reports the byte count dropped.
    This ensures we never bisect a JSON envelope that begins on a ``\\n\\n``
    boundary.
    """
    if len(text) <= _TRUNC_MAX:
        return text
    head = text[:_TRUNC_HEAD]
    tail = text[-_TRUNC_TAIL:]
    dropped = len(text) - _TRUNC_HEAD - _TRUNC_TAIL
    marker = f"\n\n[... truncated {dropped} bytes ...]\n\n"
    return head + marker + tail


def _load_prompt(kind: str) -> str:
    path = PROMPT_DIR / PROMPT_FILES.get(kind, "distill_stop.md")
    if path.exists():
        return path.read_text()
    return _FALLBACK_PROMPT


def placeholders(template: str) -> set[str]:
    return set(_PLACEHOLDER.findall(template))


def fill(template: str, **vars: str) -> str:
    """Substitute every placeholder in one pass; raise KeyError on a missing one.

    Substituted values are never rescanned, so a memory that itself contains
    `{new_text}` or `{x}` reaches the model literally.
    """
    missing = placeholders(template) - vars.keys()
    if missing:
        raise KeyError(f"prompt placeholders without values: {sorted(missing)}")
    return _PLACEHOLDER.sub(lambda m: str(vars[m.group(1)]), template)


def render(kind: str, **vars: str) -> str:
    return fill(_load_prompt(kind), **vars)


FALLBACKS = ("claude_code", "ollama")


def chain(cfg: dict) -> list[str]:
    """The primary backend, then each valid fallback once."""
    out = [cfg["distill"]["backend"]]
    for b in cfg["distill"].get("fallback") or []:
        if b not in FALLBACKS:
            log.warning("distill fallback %r ignored (allowed: %s)", b, ", ".join(FALLBACKS))
        elif b not in out:
            out.append(b)
    return out


def call(prompt: str, model: str = "haiku", timeout: int = 120, json_out: bool = False,
         backend: str | None = None) -> str | None:
    """Send an already-rendered prompt to *backend* (default: the configured one). None on failure.

    *json_out* asks Ollama for a JSON-only reply (the Claude prompts ask for it in text).
    """
    backend = backend or load_config()["distill"]["backend"]
    if backend == "passthrough":
        return None
    if backend == "ollama":
        cfg = load_config()
        return _ollama(prompt, cfg["distill"]["ollama"]["model"], cfg["embed"]["endpoint"], json_out) or None
    return run_claude(prompt, model, timeout)


def ask(kind: str, model: str | None = None, **vars: str) -> str | None:
    """render(kind, **vars) then call() with the model configured for *kind*."""
    chosen = model or load_config()["distill"]["models"].get(kind, "haiku")
    return call(render(kind, **vars), chosen)


def _ollama(prompt: str, model: str, endpoint: str, json_out: bool = False) -> str:
    """Invoke local Ollama for distillation."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 16384},
    }
    if json_out:
        body["format"] = "json"
    try:
        r = httpx.post(f"{endpoint}/api/generate", json=body, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("ollama distill failed: %s", e)
        return ""


def distill(kind: str, text: str, model: str | None = None) -> str | None:
    """Distill an event excerpt into a memory-ready summary.

    Returns a non-empty string on success, None when the backend failed (callers
    keep the events for retry rather than silently losing them), and "" for
    empty input.
    """
    if not text.strip():
        return ""
    cfg = load_config()
    if cfg["distill"]["backend"] == "passthrough":
        return _smart_truncate(text)
    chosen = model or cfg["distill"]["models"].get(kind, "haiku")
    prompt = render(kind, text=text)
    backends = chain(cfg)
    for i, b in enumerate(backends):
        out = call(prompt, chosen, json_out=True, backend=b)
        if out:
            if i:
                log.info("distill %s answered by fallback %s", kind, b)
            return out
    return None
