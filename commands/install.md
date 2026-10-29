---
description: Run the full biss-brain installer (venv, ollama model, config, launchd plist, schema migration).
argument-hint: "[--with-dmn] [--reinstall]"
---

Running biss-brain installer...

!`PR="${CLAUDE_PLUGIN_ROOT:-$(for d in ~/.claude/plugins/cache/*/biss-brain/*/; do [ -f "${d}scripts/install.py" ] && printf '%s\t%s\n' "$(basename "$d")" "${d%/}"; done | sort -V -k1,1 | tail -1 | cut -f2-)}"; CLAUDE_PLUGIN_ROOT="$PR" python3 "$PR/scripts/install.py" $ARGUMENTS`

If any steps failed, follow the hints above and re-run `/biss-brain:install`.
Pass `--with-dmn` to also install the DMN idle-maintenance daemon.
Pass `--reinstall` to force re-copy of plists and config.
