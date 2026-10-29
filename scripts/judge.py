"""Conflict judge.

Public API:
  - judge_with_project_context(new_chunk, old_chunk, same_project, cfg) -> dict
  - judge_conflict(new_text, old_text, model) -> str | None   (same-project shorthand)
  - judge_conflicts(pairs, model) -> list[str | None]        (same project, one call)

Verdicts: CONTRADICTS | EXTENDS | UNRELATED | DEFER for same-project pairs,
DIVERGES | UNRELATED | DEFER across projects. Any failure (CLI unreachable,
timeout, unparseable reply) is None — never a verdict. Callers must not act
on None.

R2 structural guard: cross-project CONTRADICTS is rewritten to DIVERGES after
parsing, unconditionally — the model can never supersede across projects.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from cli import MODEL_MAP, run_claude  # noqa: F401 — MODEL_MAP re-exported
from distill import fill

log = logging.getLogger("biss-brain.judge")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_conflict.md"
BATCH_PROMPT_PATH = PROMPT_PATH.with_name("judge_conflict_batch.md")

SAME_PROJECT_VERDICTS = {"CONTRADICTS", "EXTENDS", "UNRELATED", "DEFER"}
CROSS_PROJECT_VERDICTS = {"DIVERGES", "UNRELATED", "DEFER"}
VALID_VERDICTS = SAME_PROJECT_VERDICTS | CROSS_PROJECT_VERDICTS

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_verdict(raw: str | None) -> tuple[str | None, str, str]:
    """Return (verdict, action, reason) from a model reply; verdict None if unparseable.

    Accepts the JSON object the prompt asks for (any key/value case, optional
    code fence) or a bare verdict word.
    """
    if not raw:
        return None, "none", ""
    text = _FENCE.sub("", raw.strip()).strip()
    action, reason, word = "none", "", text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        word = str(lowered.get("verdict") or "")
        action = str(lowered.get("action") or "none")
        reason = str(lowered.get("reason") or "")
    word = word.strip().upper()
    for v in VALID_VERDICTS:
        if word.startswith(v):
            return v, action, reason
    return None, action, reason


def _judge(
    new_text: str, old_text: str, new_label: str, old_label: str, same_project: bool, model: str
) -> dict:
    try:
        template = PROMPT_PATH.read_text()
    except OSError as e:
        log.warning("judge prompt not found: %s", e)
        return {"verdict": None, "action": "none", "reason": "prompt load error"}

    prompt = fill(
        template,
        new_text=new_text,
        old_text=old_text,
        new_project_label=new_label,
        old_project_label=old_label,
        same_project="true" if same_project else "false",
    )
    raw = run_claude(prompt, model, timeout=60)
    if raw is None:
        return {"verdict": None, "action": "none", "reason": "judge call failed"}

    verdict, action, reason = parse_verdict(raw)
    if verdict is None:
        log.warning("judge: unparseable reply (%d chars); no verdict", len(raw))
        return {"verdict": None, "action": "none", "reason": "unparseable reply"}

    if not same_project:
        # ── R2 STRUCTURAL GUARD ── runs after the LLM, unconditionally.
        if verdict == "CONTRADICTS":
            log.warning(
                "R2 guard: CONTRADICTS for cross-project pair (old=%s, new=%s); rewriting to DIVERGES",
                old_label, new_label,
            )
            verdict = "DIVERGES"
            action = "write cross_links only; never superseded_by"
            reason = reason or "Cross-project divergence — supersession forbidden (R2)"
        elif verdict == "EXTENDS":
            verdict = "UNRELATED"
    elif verdict == "DIVERGES":
        # DIVERGES is a cross-project verdict; same-project it carries no action.
        verdict = "UNRELATED"

    return {"verdict": verdict, "action": action, "reason": reason}


def judge_with_project_context(
    new_chunk: dict,
    old_chunk: dict,
    same_project: bool,
    cfg: dict,
    model: str | None = None,
) -> dict:
    """Judge two chunks with both project labels in the prompt.

    Args:
        new_chunk / old_chunk: dicts with at least {"text", "project"}.
        same_project: True when both chunks share the same project hash.
        cfg: loaded config (supplies distill.models.conflict_judge).

    Returns {"verdict": str | None, "action": str, "reason": str}.
    """
    chosen = model or cfg.get("distill", {}).get("models", {}).get("conflict_judge", "haiku")
    if same_project:
        new_label = old_label = _project_label(new_chunk.get("project", "unknown"))
    else:
        new_label = _project_label(new_chunk.get("project", "unknown"))
        old_label = _project_label(old_chunk.get("project", "unknown"))
    return _judge(
        new_chunk.get("text", ""), old_chunk.get("text", ""),
        new_label, old_label, same_project, chosen,
    )


def judge_conflict(new_text: str, old_text: str, model: str = "haiku") -> str | None:
    """Same-project verdict for *new_text* vs *old_text*, or None on any failure."""
    return _judge(new_text, old_text, "(same project)", "(same project)", True, model)["verdict"]


def judge_conflicts(pairs: list[tuple[str, str]], model: str = "haiku") -> list[str | None]:
    """Same-project verdicts for [(new_text, old_text), …] in one call.

    One pair goes through judge_conflict. An entry is None when its verdict is
    missing or unparseable, and every entry is None when the call fails.
    """
    if len(pairs) <= 1:
        return [judge_conflict(n, o, model) for n, o in pairs]
    try:
        template = BATCH_PROMPT_PATH.read_text()
    except OSError as e:
        log.warning("judge batch prompt not found: %s", e)
        return [None] * len(pairs)
    # A JSON array, so a memory cannot forge another pair's header.
    body = json.dumps(
        [{"pair": i, "old": old, "new": new} for i, (new, old) in enumerate(pairs, 1)],
        ensure_ascii=False, indent=1,
    )
    raw = run_claude(fill(template, pairs=body), model, timeout=90)
    out: list[str | None] = [None] * len(pairs)
    if not raw:
        return out
    text = _FENCE.sub("", raw.strip()).strip()
    try:
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        log.warning("judge batch: unparseable reply (%d chars); no verdicts", len(raw))
        return out
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("pair")) - 1
        except (TypeError, ValueError):
            continue
        verdict = parse_verdict(json.dumps(item))[0]
        if 0 <= i < len(pairs) and verdict in SAME_PROJECT_VERDICTS:
            out[i] = verdict
    return out


def _project_label(project_hash: str) -> str:
    """Human-readable label from the project's project_path.txt, else the hash."""
    try:
        from common import BRAIN_ROOT  # noqa: PLC0415
        label_file = BRAIN_ROOT / project_hash / "project_path.txt"
        if label_file.exists():
            return label_file.read_text().strip() or project_hash
    except Exception:  # noqa: BLE001
        pass
    return project_hash
