---
description: Share a memory of this project with every project (or take it back with --off).
argument-hint: "[--off] <chunk_id> [<chunk_id> ...]"
disable-model-invocation: true
---

Sharing **$ARGUMENTS**:

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/share.py" $ARGUMENTS`
