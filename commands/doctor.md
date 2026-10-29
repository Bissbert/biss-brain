---
description: Run the biss-brain preflight checker to verify all dependencies are in place.
argument-hint: (no arguments)
---

Running biss-brain preflight checks...

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -x "${d}scripts/_run.sh" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" "$PR/scripts/_run.sh" "$PR/scripts/doctor.py"`

If any checks failed, follow the remediation hints above. Re-run `/biss-brain:doctor` after fixing.
