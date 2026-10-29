"""Transcript tail → source-tagged text for the distiller.

The raw JSONL tail mixes what the user typed, what the assistant wrote, tool
output, web pages, harness records (system prompts, attachments) and memories
biss-brain injected earlier. Distilling it as one blob launders all of it into
"the user said". This module classifies by *block type*, not message role —
tool results arrive inside user-role messages — and renders one segment per
block:

    [user] text the user typed
    [claude] text the assistant wrote, or a compact tool call
    [web] a WebFetch / WebSearch / MCP result
      continuation lines are indented, so no content line can start a segment

Only `user` and `assistant` entries are read; every other record type
(api-request-blob, attachment, system, …) is dropped. Also dropped: isMeta and
compact-summary entries, <system-reminder> blocks, earlier <brain-memory>
injections, biss-brain's own MCP calls and results, thinking, and anything
inside <private>…</private>. Literal source tags inside content are defused.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import provenance
from provenance import tool_source

log = logging.getLogger("biss-brain.transcript")

# Display tag per source; the distiller echoes these on each bullet.
TAG = dict(user="user", file="file", tool="tool", web="web", assistant="claude")
SOURCE_OF_TAG = {v: k for k, v in TAG.items()} | {"assistant": "assistant"}

_PRIVATE = re.compile(r"<private\b[^>]*>.*?(?:</private\s*>|\Z)", re.I | re.S)
_REMINDER = re.compile(r"<system-reminder\b[^>]*>.*?(?:</system-reminder\s*>|\Z)", re.I | re.S)
_BRAIN = re.compile(r"<brain-memory\b[^>]*>.*?(?:</brain-memory\s*>|\Z)", re.I | re.S)
# Harness-generated blocks that appear in user-role strings: tool output, not user words.
_HARNESS = re.compile(
    r"<(task-notification|local-command-stdout|local-command-stderr|bash-stdout|bash-stderr)\b[^>]*>"
    r"(.*?)(?:</\1\s*>|\Z)",
    re.I | re.S,
)
_FORGED = re.compile(r"\[(\s*)(user|claude|assistant|file|tool|web|imported|system)\b([^\]\n]*)\]", re.I)
_OWN_MCP = "mcp__biss-brain__"
_TOOL_INPUT_CHARS = 300


@dataclass(frozen=True)
class Segment:
    source: str  # raw provenance source
    text: str


def strip_private(text: str) -> str:
    """Drop <private>…</private> (case-insensitive; an unclosed tag drops to the end)."""
    return _PRIVATE.sub("", text)


def defuse(text: str) -> str:
    """Neutralise literal source tags so content cannot forge a segment or bullet tag."""
    return _FORGED.sub(lambda m: f"({m.group(1)}{m.group(2)}{m.group(3)})", text)


def _clean(text: str) -> str:
    text = strip_private(text)
    text = _REMINDER.sub("", text)
    text = _BRAIN.sub("", text)
    return defuse(text).strip()


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _user_string(text: str) -> list[Segment]:
    out: list[Segment] = []
    rest = _HARNESS.sub(lambda m: out.append(Segment("tool", m.group(2))) or "", text)
    return [Segment("user", rest), *out]


def segments(jsonl: str) -> list[Segment]:
    """Classify a transcript JSONL tail into source-tagged segments (in order)."""
    return [s for segs in line_segments(jsonl.splitlines()) for s in segs]


def line_segments(lines: list[str]) -> list[list[Segment]]:
    """The source-tagged segments of each JSONL line, in order (one list per line)."""
    names: dict[str, str] = {}  # tool_use id → tool name
    per_line: list[list[Segment]] = []
    for line in lines:
        out: list[Segment] = []
        per_line.append(out)
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("type") not in ("user", "assistant"):
            continue
        if d.get("isMeta") or d.get("isCompactSummary"):
            continue
        content = (d.get("message") or {}).get("content")
        if d["type"] == "user":
            if isinstance(content, str):
                out += _user_string(content)
                continue
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    out += _user_string(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    name = names.get(b.get("tool_use_id", ""), "")
                    if name.startswith(_OWN_MCP):
                        continue
                    out.append(Segment(tool_source(name), _block_text(b.get("content"))))
            continue
        for b in content if isinstance(content, list) else [{"type": "text", "text": content or ""}]:
            if not isinstance(b, dict):
                continue
            t = b.get("type") or ""
            if t == "text":
                out.append(Segment("assistant", b.get("text", "")))
            elif t in ("tool_use", "server_tool_use"):
                name = b.get("name", "")
                names[b.get("id", "")] = "WebSearch" if t == "server_tool_use" else name
                if name.startswith(_OWN_MCP):
                    continue
                args = json.dumps(b.get("input", {}), ensure_ascii=False)[:_TOOL_INPUT_CHARS]
                out.append(Segment("assistant", f"→ {name} {args}"))
            elif t == "thinking" or t == "redacted_thinking":
                continue
            elif t.endswith("_tool_result"):
                # server-side results (web_search_tool_result, …) are third-party content
                src = "web" if "web" in t else "tool"
                out.append(Segment(src, _block_text(b.get("content")) or json.dumps(b.get("content", ""))[:2000]))
    return [[c for c in (Segment(s.source, _clean(s.text)) for s in out) if c.text] for out in per_line]


def _blocks(segs: list[Segment]) -> list[str]:
    blocks = []
    for s in segs:
        first, *rest = s.text.splitlines() or [""]
        blocks.append("\n".join([f"[{TAG[s.source]}] {first}", *(f"  {ln}" for ln in rest)]))
    return blocks


def sources_in(text: str) -> list[str]:
    """Sorted raw sources whose tags start a segment in rendered *text*."""
    return sorted({SOURCE_OF_TAG[m] for m in re.findall(r"^\[(\w+)\] ", text, re.M) if m in SOURCE_OF_TAG})


def line_windows(
    lines: list[tuple[int, list[Segment]]], max_chars: int, max_windows: int
) -> list[tuple[str, int]]:
    """Pack whole lines' segments, oldest first, into windows of at most *max_chars*.

    *lines* is [(byte offset after the line, its segments)]. Returns
    [(window text, offset after its last line)], at most *max_windows*; the
    lines after the last one are left for the next pass. A line longer than
    a window is cut to its last *max_chars*. Lines without text extend the
    window before them.
    """
    out: list[tuple[str, int]] = []
    cur: list[str] = []
    size, end = 0, None
    for off, segs in lines:
        if not segs:
            end = off
            continue
        text = "\n".join(b[-max_chars:] for b in _blocks(segs))[-max_chars:]
        if cur and size + len(text) + 1 > max_chars:
            out.append(("\n".join(cur), end))
            if len(out) == max_windows:
                return out
            cur, size = [], 0
        cur.append(text)
        size += len(text) + 1
        end = off
    if cur:
        out.append(("\n".join(cur), end))
    elif end is not None and out:
        out[-1] = (out[-1][0], end)
    return out


def render(segs: list[Segment], max_chars: int) -> str:
    """Tagged text, newest segments kept when over *max_chars*."""
    blocks = _blocks(segs)
    kept: list[str] = []
    size = 0
    for b in reversed(blocks):
        if size + len(b) + 1 > max_chars:
            if not kept:
                kept.append(b[-max_chars:])
            break
        kept.append(b)
        size += len(b) + 1
    return "\n".join(reversed(kept))


def excerpt(jsonl: str, max_chars: int) -> tuple[str, list[str]]:
    """(rendered excerpt, sorted raw sources present in it)."""
    segs = segments(jsonl)
    text = render(segs, max_chars)
    return text, sources_in(text)


# ── distiller output → per-source groups ─────────────────────────────────

_BULLET = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)?\[(\w+)\]\s*(.*)$")
_HEADING = re.compile(r"^\s*#{1,6}\s")
TRUSTED = provenance.TRUSTED_SOURCES


def _present_and_floor(present: list[str]) -> tuple[list[str], str]:
    """Known sources present, and the source for an untagged or unknown-tagged line.

    The floor is the least-trusted source present, but never `user`: a line the
    model did not attribute was not typed by the user.
    """
    from provenance import DEFAULT_TRUST  # noqa: PLC0415

    present = [p for p in present if p in DEFAULT_TRUST] or ["tool"]
    floor = min(present, key=lambda s: DEFAULT_TRUST[s])
    return present, "assistant" if floor == "user" else floor


def split_summary(summary: str, present: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Split a tagged distiller summary into {"trusted"|"low": [(source, line), …]}.

    A bullet's tag is only honoured if that source was present in the batch;
    untagged or unknown-tagged lines take the least-trusted source present
    (never `user`). Headings are copied into each group that has lines under them; continuation
    lines follow their bullet. Tags are removed from the stored text.
    """
    present, floor = _present_and_floor(present)
    groups: dict[str, list[tuple[str, str]]] = {"trusted": [], "low": []}
    heading = None
    emitted: dict[str, str | None] = {"trusted": None, "low": None}
    last = None
    for line in summary.splitlines():
        if not line.strip() or line.strip() in ("---", "***", "___"):
            continue
        if _HEADING.match(line):
            heading = line
            continue
        if last is not None and line[:1].isspace() and not _BULLET.match(line):
            groups[last].append((groups[last][-1][0], line))
            continue
        m = _BULLET.match(line)
        if m:
            tag = m.group(2).lower()
            src = SOURCE_OF_TAG.get(tag)
            src = src if src in present else floor
            text = (m.group(1) or "") + m.group(3)
        else:
            src, text = floor, line
        g = "trusted" if src in TRUSTED else "low"
        if heading and emitted[g] != heading:
            groups[g].append((src, heading))
            emitted[g] = heading
        groups[g].append((src, text))
        last = g
    return groups


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
# Distiller salience for web/tool text is capped: that text could ask for a high score.
LOW_SALIENCE_CAP = 5
_LEAD = re.compile(r"^\s*(?:(?:[-*]|\d+\.)\s+)?(?:\[(?:user|claude|assistant|file|tool|web)\]\s*)?", re.I)


def _salience(v) -> int | None:
    try:
        return max(1, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def _bullets(text: str) -> list | None:
    """The bullet list of a JSON reply, or its complete bullet objects if it does not parse.

    None when there is neither: the reply is not the JSON the prompt asked for.
    """
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if isinstance(obj, dict) and isinstance(obj.get("bullets"), list):
            return obj["bullets"]
    except ValueError:  # json.JSONDecodeError is a ValueError, as is a missing brace
        pass
    out: list = []
    start = text.find("[", text.find('"bullets"'))
    dec = json.JSONDecoder()
    i = text.find("{", start + 1) if start >= 0 else -1
    while i >= 0:
        try:
            b, end = dec.raw_decode(text, i)
        except ValueError:
            break
        out.append(b)
        i = text.find("{", end)
    log.warning("distill reply is not valid JSON; kept %d complete bullet(s)", len(out))
    return out or None


def split_output(
    raw: str, present: list[str], strict: bool = False
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]] | None:
    """Split a distiller reply into trust groups, plus the salience of each group.

    The reply is the JSON the distill prompts ask for:
    {"bullets": [{"source", "section"?, "text", "salience"}]}. A bullet's source
    is honoured only if it was present in the segment; otherwise it takes the
    floor (split_summary). Sections become headings, contiguous per group. A
    group's salience is its highest bullet salience (at most LOW_SALIENCE_CAP
    for the low group).

    A reply that starts like that JSON but does not parse (cut off, or a
    stray character) keeps only its complete bullet objects; it is never
    stored as text. Any other reply is read as the older tagged-markdown
    format, with no salience; `[no findings]` is empty.

    *strict* (any backend that was sent a JSON prompt): None, a failed
    distill, for a reply that is neither that JSON, nor complete bullets
    salvaged from it, nor `[no findings]`.
    """
    text = _FENCE.sub("", raw.strip()).strip()
    groups: dict[str, list[tuple[str, str]]] = {"trusted": [], "low": []}
    if text.strip("`").strip().lower() == "[no findings]":
        return groups, {}
    if not (text.startswith("{") or '"bullets"' in text):
        return None if strict else (split_summary(raw, present), {})
    bullets = _bullets(text)
    if bullets is None:
        return None if strict else (groups, {})

    present, floor = _present_and_floor(present)
    by_group: dict[str, dict[str, list[tuple[str, str]]]] = {"trusted": {}, "low": {}}
    salience: dict[str, int] = {}
    for b in bullets:
        if not isinstance(b, dict):
            continue
        body = _LEAD.sub("", " ".join(str(b.get("text") or "").split()))
        if not body:
            continue
        src = SOURCE_OF_TAG.get(str(b.get("source") or "").strip().lower())
        src = src if src in present else floor
        g = "trusted" if src in TRUSTED else "low"
        section = " ".join(str(b.get("section") or "").split())
        by_group[g].setdefault(section, []).append((src, f"- {body}"))
        sal = _salience(b.get("salience"))
        if sal is not None:
            salience[g] = max(salience.get(g, 0), min(sal, LOW_SALIENCE_CAP) if g == "low" else sal)
    for g, sections in by_group.items():
        for section, lines in sections.items():
            if section:
                groups[g].append((lines[0][0], f"### {section}"))
            groups[g].extend(lines)
    return groups, salience
