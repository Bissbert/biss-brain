# Changelog

## 0.8.1 — 2026-10-29

### Fixed

- Distill stopped working when a Claude Code session started before an update kept running an
  older flush daemon. A capture from a newer plugin copy now stops a daemon started from an older
  one and takes over; queued events stay on disk and drain. `/biss-brain:doctor` flags an
  outdated daemon.
- A NUL byte in a transcript window crashed the distill call (`embedded null byte`). It is
  dropped from the prompt, and a call that cannot start counts as a CLI failure.
- A failed `claude -p` call logs its reason. With JSON output the CLI reports errors on stdout,
  so the log showed an empty `stderr=` before.
- The UI's status page read keys the server never sent, so "Last complete", the error time and
  both job tiles were always empty or ✗.
- The nightly dream pass records each project's last completion or error in `dream_state`; only
  the optional DMN job wrote it before.

## 0.8.0 — 2026-10-29

Public release. The repo is now its own marketplace:
`/plugin marketplace add Bissbert/biss-brain`, then `/plugin install biss-brain@bissbert`.

### Added

- The Python environment builds itself. After an install or an update, the first hook starts
  `uv sync` in the background (`scripts/build_venv.sh`, one build per plugin copy) and the
  session-start message says so. The nightly jobs wait for the build instead.
- First-run hints at session start: when uv is missing, while the environment builds, and when
  `/biss-brain:install` has not been run yet.
- A stable launchd runner, `~/.claude/brain/bin/run`. The jobs no longer point at one versioned
  plugin folder, so they keep working after an update.
- `.claude-plugin/marketplace.json`, an MIT `LICENSE`, and README graphics.

### Changed

- The README is now for users: install, privacy, commands, configuration. The developer material
  moved to `docs/internals.md`.
- The slash commands and `/biss-brain:install` find the plugin under any marketplace.
- `/biss-brain:doctor` checks for a finished build (`.venv/.biss-ready`) and flags a launchd job
  still pinned to one plugin folder.
- `uninstall.sh` also removes the idle-time job, the runner and a leftover build lock.

### Upgrading

Run `/biss-brain:install` once after updating. It switches the launchd jobs to the runner; until
then they keep pointing at the old version's folder.

## 0.7.3 — 2026-10-29

### Changed

- The README's install section now describes the marketplace install (`/plugin install`, then
  `/biss-brain:install`) and its requirements, instead of a path in the author's checkout.
- Removed the unused `launchd/*.plist` files. `/biss-brain:install` writes the jobs itself, and
  the copies in the repo held one machine's absolute paths.
- Docs no longer name the author's home directory, machine or other projects.

## 0.7.2 — 2026-10-28

### Fixed

- Distillation works again for a Claude Code login kept in the keychain. Every `claude -p` call
  used `--bare`, which reads auth only from `ANTHROPIC_API_KEY`, so each call came back "Not
  logged in" and windows sat in the retry queue. The child now runs with `--setting-sources ""`,
  `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence` and `--tools ""`
  instead: no hooks, plugins, MCP servers or tools load, so it still cannot recurse into capture.
- The `claude -p` child gets an empty stdin. It used to inherit the caller's, and `-p` appends
  piped stdin to the prompt.

## 0.7.1 — 2026-10-28

### Fixed

- `/biss-brain:install` no longer rewrites `config.toml` when it records `[distill] claude_bin`.
  It used to re-serialise the whole file, which dropped every comment and turned nested tables
  such as `[distill.models]` into invalid TOML, so every reader failed to load the config. It
  now edits that one line and checks that the result parses before writing.

## 0.7.0 — 2026-10-26

v0.7 hardens what v0.6 built: it measures it, makes it safe against poisoned content, gives
memories event time, adds a curated core tier, and makes capture cheap. The plan is
`docs/plans/v0.7-upgrade.md`. Each phase's record, eval numbers included, is in
`docs/plans/v0.7-progress.md`.

### Upgrade

Start fresh: `scripts/fresh_start.py` archives `~/.claude/brain` to
`~/.claude/brain-archive/brain-<ts>/` and carries over `config.toml` and core memory. See
"Upgrading to v0.7" in the README for the order of steps. `scripts/run_v07_migrations.sh` is the
in-place alternative; it writes a backup first.

### Behaviour changes

Things that work differently after the upgrade, with the setting that restores the old
behaviour where there is one.

- **The store refuses a different embed model.** `store.json` records the model, dim and task
  prefixes. If `config.toml` differs, the hooks inject nothing, flush leaves the queue alone,
  dream only decays, and recall, `/remember` and the MCP search tools report an error. Restore
  the `[embed]` settings, or start a new store.
- **Capture scope.** By default nothing is captured or injected in `~`, `~/Downloads`,
  `~/Library`, `~/.claude` or `/tmp` (`[scope] exclude`). A `.biss-brain-ignore` file or
  `.claude/brain.toml` with `enabled = false` turns a project off.
- **One project key per repository.** Worktrees and subdirectories of a repo share the main
  checkout's store. Before, each worktree had its own.
- **Capture by segment.** Stop no longer distills. Events are queued and the daemon distills a
  session segment once it ends (SessionEnd, PreCompact, 10 min without a Stop, or 20 Stops).
  Memories land minutes later than before, at about one LLM call per segment.
- **PreCompact returns at once** (hook timeout 120 s → 10 s). The restore after compaction uses
  a digest of the transcript until the distilled summary lands, and only this session's summary.
- **Provenance.** Tool and web content is stored with its source, never auto-injected, ranked at
  × 0.6 in search, and can't supersede or merge into first-party memory. Low-trust text that
  reads like an instruction is quarantined. `/remember` always writes `user` provenance.
- **Injected memories are data.** They arrive in escaped `<brain-memory>` blocks.
- **Contradictions keep history.** A contradicted memory gets `invalid_at` and drops out of
  current results, but `as_of` and `include_history` still find it. The newer fact by event time
  always wins.
- **Ranking.** One ranking path for every reader: hybrid BM25 + dense, then relevance ×
  decay ^ `decay_weight`. `decay_weight` is 0.05 (was 0.1). Recency alone no longer reorders
  results.
- **Prompt injection gate** (still off by default, `retrieve.user_prompt_inject`): cosine ≥ 0.6
  or BM25 ≥ 7, top 1, 6,000 characters. It has an embed budget, a deadline, a per-session ledger
  and `/biss-brain:inject off`. The `user_prompt_inject_*` keys are renamed to `inject_*`; the
  old names are still read.
- **RIF is off** (`[rif] enabled = false`). It cost 0.105 MRR on the real suite and about
  170 ms per read.
- **No usage boost in nightly decay** (`[hygiene] decay_usage_boost = 0.0`; 0.2 restores v0.6).
- **Reconsolidation** merges only memories that were used inside the labile window and that the
  new text revises by ≥ 20 % new content terms (`reconsolidation.revise_min_novelty`). A
  restatement is folded in without an LLM call.
- **Failures are retried.** A distill reply that is not valid JSON is a failure, retried with
  backoff, never stored as text. When the embedder is down, chunks are stored without a vector
  and backfilled later (`[embed] store_without_vector`).
- **Distill models** resolve to `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`.
  Every CLI call runs the resolved binary with `--bare` and no tools.

### Added

- Core memory: `~/.claude/brain/core/` (its own git repo), `/biss-brain:core`,
  `/remember --core`, dream proposals, and injection at SessionStart and after compaction.
- `as_of` and `include_history` on recall and MCP search.
- `/biss-brain:inject on|off|status`, and `inject_metrics.jsonl` (no prompt text).
- SessionEnd hook.
- `scripts/fresh_start.py`, `scripts/migrate_v07.py`, `scripts/run_v07_migrations.sh`.
- Doctor checks: store fingerprint (a mismatch fails), schema version, FTS coverage, model and
  table vector dims, the daemons' claude CLI, launchd plists on another plugin version, the last
  dream pass, pending work, scope, core memory and the prompt hook.
- Eval harness (`scripts/eval/`) and the ablation in `docs/eval/v0.7-ablation.md`.

### Fixed

- LanceDB filters are validated, and no f-string filters remain. Updates are project-scoped,
  and cross-project ids are rejected.
- Distill prompts reach the CLI fully rendered (salience and merge used to send literal
  placeholders).
- The conflict judge's prompt and parser agree, and a failed judge call is retried rather than
  read as a verdict.
- The FTS index is refreshed after every write.
- The scrubber covers `sk-ant-`, `github_pat_`, `glpat-`, PEM blocks and `user:pass@` URLs.
