---
description: Search this project's brain for relevant past findings.
argument-hint: <query>
---

The user asked the brain to recall information about: **$ARGUMENTS**

Brain search results:

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/recall.py" "$ARGUMENTS"`

Use these as grounding context for the user's task. If results are empty, tell the user the brain has no matching memory yet.
