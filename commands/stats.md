---
description: Show biss-brain memory statistics for the current project.
---

Brain statistics for the current project:

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/stats.py"`

Summarise these stats for the user briefly.
