---
description: Save an explicit user memory into the project brain.
argument-hint: "[--kind decision|note|code|task] [--pin] [--core [--shared|--global] [--section decisions|conventions|lessons|pinned]] <text>"
disable-model-invocation: true
---

Saving memory: **$ARGUMENTS**

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/remember.py" --source user $ARGUMENTS`
