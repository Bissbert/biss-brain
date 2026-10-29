---
description: Mark a brain chunk as superseded so it stops appearing in retrieval.
argument-hint: <chunk_id>
---

Marking chunk **$ARGUMENTS** as superseded:

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/forget.py" $ARGUMENTS`
