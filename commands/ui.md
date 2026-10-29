---
description: Launch the biss-brain UI inspector for this project.
argument-hint: "[--stop]"
---

Launching the biss-brain UI inspector:

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/launch_ui.py" $ARGUMENTS`

Set `BISS_BRAIN_DEV=1` in the environment to enable uvicorn auto-reload (watches `ui/` and `scripts/`) for development.
