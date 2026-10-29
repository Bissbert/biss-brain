---
description: Show, review and curate core memory (the notes injected in full every session).
argument-hint: "[show | review | accept <id> [--layer private|global] | reject <id> | remove <m:id> | shared on|off | export]"
disable-model-invocation: true
---

Core memory: **$ARGUMENTS**

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/core_cmd.py" $ARGUMENTS`

Show the output above to the user as-is. For `review`, do not accept or reject proposals yourself: the user runs `/biss-brain:core accept <id>` or `/biss-brain:core reject <id>`.
