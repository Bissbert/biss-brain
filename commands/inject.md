---
description: Turn prompt-time memory injection on or off for this project, and show its latency and outcomes.
argument-hint: "[status | on | off]"
disable-model-invocation: true
---

Prompt injection: **$ARGUMENTS**

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/inject_cmd.py" $ARGUMENTS`

Show the output above to the user as-is.
