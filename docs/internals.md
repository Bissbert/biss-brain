# biss-brain internals

How biss-brain works under the hood: the stack, file layout, the capture and
flush daemon, the nightly dream pass, which mechanisms are on and why, and
known limits. For installing and using it, see the [README](../README.md).
The design research and plans behind it are indexed in [docs/README.md](README.md).

## Stack

- **Vector store:** LanceDB (embedded, file-backed, native hybrid BM25 + vector)
- **Embeddings:** Ollama `nomic-embed-text` (768-dim) by default; set `[embed] model`/`dim` in `config.toml`. Vectors are unit-normalised and every gate uses cosine
- **Distillation:** `claude -p` headless — Haiku for routine, Sonnet for `PreCompact`
- **Metadata:** SQLite (canonical state for `superseded_by`, `pinned`, `retrieval_count`)
- **MCP:** FastMCP stdio
- **Storage:** `~/.claude/brain/<project-hash>/`

## Layout

```
biss-brain/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json            # the repo is its own marketplace ("bissbert")
├── hooks/hooks.json                # Stop, SubagentStop, PreCompact, SessionEnd, SessionStart, UserPromptSubmit
├── .mcp.json                       # registers the MCP server
├── commands/                       # /biss-brain:* slash commands
├── prompts/                        # distillation prompts (edit freely)
├── scripts/
│   ├── _run.sh                     # launcher: scope check, builds .venv on first use, runs python
│   ├── build_venv.sh               # uv sync --frozen under a per-copy lock; writes .venv/.biss-ready
│   ├── launchd_run.sh              # copied to ~/.claude/brain/bin/run; launchd jobs run it
│   ├── install.py                  # /biss-brain:install
│   ├── doctor.py                   # /biss-brain:doctor
│   ├── common.py                   # config, paths, schema
│   ├── capture.py                  # hook entry; writes to the queue
│   ├── flush.py                    # background daemon; segments, distill, embed, index
│   ├── distill.py, cli.py          # claude -p / ollama backends
│   ├── inject_context.py           # SessionStart, compact restore, prompt injection
│   ├── retrieve.py, rerank.py      # hybrid search
│   ├── core.py, core_cmd.py        # core memory
│   ├── recall.py, remember.py, forget.py, stats.py   # command backends
│   ├── dream.py                    # nightly consolidation (Stage 1 + Stage 2)
│   └── dmn.py                      # optional idle-time maintenance
├── mcp/server.py                   # FastMCP tools
└── ui/                             # FastAPI web UI
```

## Storage

```
~/.claude/brain/
  config.toml                       # global config
  active_root                       # the live plugin copy, recorded by _run.sh
  bin/run                           # launchd runner (from scripts/launchd_run.sh)
  store.json                        # schema version and embed fingerprint (v0.7)
  flush.pid                         # daemon lock
  flush.log                         # daemon stdout/stderr
  core/                             # core memory, its own git repo (auto-committed)
    _global.md                      # every project
    <repo>.md                       # this repo, private (frontmatter: repo path, remote)
    <repo>.proposed.md              # pending proposals, git-ignored
  core.sqlite                       # core item versions and the proposal queue
  scope.cache                       # [scope] rules for _run.sh (written by scope.py)
  <project-hash>/                   # sha256(main checkout of the repo)[:12]; realpath outside a repo
    queue/                          # pending events drained by flush.py
    lancedb/chunks.lance/           # vector + FTS index; versions older than 10 min pruned on each write, index rebuilt nightly
    meta.sqlite                     # chunk metadata, superseded, pinned, counts
```

## Operating model

`capture.py` spawns `flush.py` on demand — the flush daemon is started lazily on the first captured event and exits after `idle_exit_seconds` of empty queues. No persistent service is needed for capture and indexing.

### Nightly dream pass

`scripts/dream.py` runs nightly at 03:00 local time via a launchd agent (`ch.bissbert.biss-brain.dream`) that `/biss-brain:install` writes to `~/Library/LaunchAgents/`. It supersedes the standalone decay job — decay is now Stage 1c of the dream pass. The dream pass is **additive**: it never deletes data, only adds new chunks and updates metadata.

| Stage | What it does | LLM cost |
|---|---|---|
| **1a Backlink refresh** | Re-runs A-MEM-style top-K linking on chunks the pass hasn't visited yet. Picks up neighbours that didn't exist at insert time (HippoRAG-style synonym edges). | None (vector search only) |
| **1b Conflict sweep** | For each chunk added in the last `candidate_window_hours`, the existing `judge.py` runs against its top-similar neighbour. `CONTRADICTS` → marks the older chunk superseded. `EXTENDS` → appends to `related_ids`. | Up to 1 Haiku call per new chunk |
| **1c Decay** | Recomputes `decay_score` for every chunk from age and pin (plus `decay_usage_boost` × retrievals, off by default). | None |
| **1d Compact** | Rebuilds the full-text index in one piece and prunes table versions older than ten minutes. Rows do not change. | None |
| **2a Session arcs** | Groups raw chunks by `session_id`, distills each session into a `kind='arc'` chunk at `consolidation_round=1`. Sets `parent_id` on the originals so the UI can collapse / expand. Skips sessions younger than `session_cooldown_hours`. | One Haiku call per qualifying session |
| **2b Daily summary** | (opt-in) Rolls up arcs by calendar day into `kind='daily_summary'` at `consolidation_round=2`. | One Sonnet call per qualifying day |
| **2c Core proposals** | Queues short, trusted, often-recalled decisions, lessons, schemas and pinned chunks as core-memory proposals. Never writes core. | None |

Retrieval can opt into "summary first" via `[dream] prefer_summaries_in_recall = true`. `inject_context.py` always prefers `arc` / `daily_summary` over raw `stop` chunks at session start, on the basis that summaries are higher-density.

Manual usage:
```bash
.venv/bin/python scripts/dream.py                       # all stages, all projects
.venv/bin/python scripts/dream.py --stage 1             # cheap pass only
.venv/bin/python scripts/dream.py --project <hash>      # one project
.venv/bin/python scripts/dream.py --dry-run             # log what would change
```

To uninstall the dream job:
```bash
launchctl unload ~/Library/LaunchAgents/ch.bissbert.biss-brain.dream.plist
rm ~/Library/LaunchAgents/ch.bissbert.biss-brain.dream.plist
```

## How the daemon works

1. A hook fires (Stop, SubagentStop, PreCompact, SessionEnd) → `capture.py` writes a small event
   file to the project's `queue/` and exits. The event holds the transcript's byte offset and the
   scrubbed last assistant message; the transcript is not parsed in the hook.
2. `capture.py` also ensures `flush.py` is running (spawned detached on first event, holds an `flock` on `flush.pid`).
3. `flush.py` polls every project's queue at 1 Hz and groups events by transcript (one session
   segment). A segment closes when:
   - a SessionEnd or PreCompact event arrives, **or**
   - no Stop arrived for `segment_idle_seconds` (10 min), **or**
   - `segment_max_events` Stops are queued.

   A subagent's own transcript closes `subagent_stop_seconds` after it stops.
4. Flush = read the transcript from the segment cursor, oldest first, up to the first boundary
   event → tag by source, strip `<private>`, scrub → pack whole JSONL lines into windows (usually
   one per segment). For each window: one distill call, which returns JSON bullets with source and
   salience → embed → upsert into LanceDB and SQLite → move the cursor (`~/.claude/brain/cursors/`)
   past the window and delete the events it finished. Then one batched conflict-judge call for the
   new chunks. A pass covers at most `raw_tail_bytes` and `segment_max_windows` windows; the rest
   is distilled in the next pass, never skipped. If the transcript is gone, the last assistant
   messages are distilled instead, tagged `[claude]`.
   With `[distill] fallback = ["ollama"]` (or `"claude_code"`), an empty reply from the primary
   backend is retried on the next backend in the chain before the pass counts as failed.
   A failed distill or embed, or a reply that is not the JSON the prompt asked for, ends the pass:
   what was stored stays, and the segment is retried from the failed window after
   `retry_base_seconds` (60 s), doubling up to `retry_max_seconds` (1 h). Only a window whose
   reply is invalid `skip_invalid_after` (5) times in a row is skipped, and recorded in
   `<project>/pending/skipped.jsonl`. The retry state (`<project>/pending/`, counts, times and
   byte ranges only) is what doctor reports.
   If only the embedder is down, the distilled chunks are stored without a vector
   (`[embed] store_without_vector`, on by default): keyword search finds them, and the daemon
   embeds, links and judges them later; while the embedder stays down it probes once per
   `backfill_interval_seconds` (60 s). A chunk the embedder rejects while it is up keeps no vector
   and is listed in `<project>/pending/embed_skip.json`. A segment or embed that waits for a retry does not keep the
   daemon running; the next capture or the nightly dream starts it again.
5. PreCompact does not wait for the distill. It also writes `sessions/<sid>.compact.json`, the
   last `compact_digest_chars` of the conversation (only `[user]` and `[claude]` segments, no
   file, tool or web text; private-stripped, scrubbed, mode 0600). SessionStart(compact) restores
   this session's summary when it is already stored, else the digest. After a digest restore, the
   first prompt after the summary lands injects it once (`[retrieve] compact_late_restore`, on by
   default and independent of `user_prompt_inject`).
6. The daemon never exits while a segment is open; it exits after `idle_exit_seconds` with
   nothing queued but segments backing off after a failure, and respawns on the next event. A
   backing-off segment is retried by the next daemon, which the next capture starts.

## Brain mechanisms

Each mechanism was ablated in v0.7 against a real and a synthetic question suite. A mechanism
stays on only if it improves MRR, or costs nothing measurable. Numbers and method are in
[docs/eval/v0.7-ablation.md](eval/v0.7-ablation.md).

| Mechanism | Default | Why |
|---|---|---|
| Hybrid retrieval (BM25 + dense) | on | Real suite MRR 0.764 vs 0.693 dense only (CI crosses 0); kept because BM25 is the fallback when the embedder is slow |
| Decay in ranking | on | Neutral, free |
| Provenance rank multiplier | on | Safety: on the synthetic suite, 0 vs 11 poisoned top-1 hits without it |
| Reconsolidation | on, gated | Not measured. Merges only used memories that new text revises; a restatement is folded without an LLM call |
| STDP edges, backlinks, DMN | on | Feed `related_to` only, not ranking; cheap |
| RIF | **off** (`[rif] enabled`) | Real suite ΔMRR −0.105, ~170 ms per read, erased nightly by decay |
| Usage boost in decay | **off** (`[hygiene] decay_usage_boost`) | No gain on either suite |
| Summary first, cross-encoder rerank, daily summary | opt-in | Summary first cost −0.125 on the synthetic suite; rerank and 2b not measured |
| Conflict sweep, session arcs, global promotion | on | Not measurable on the frozen suites; your call |

## What works

- Hybrid BM25 + vector retrieval via LanceDB RRF, with `bge-reranker-v2-m3` second-stage rerank (cross-encoder fallback if the model can't load)
- A-MEM-style bidirectional backlinks written on insert; `related_to` MCP tool reads them in O(1)
- Conflict judge (`judge.py`) returning `CONTRADICTS` / `EXTENDS` / `UNRELATED`, run on insert (one batched call per segment) and in the nightly dream pass
- Secret scrubbing (regex set + Shannon entropy second pass) before embedding
- Nightly **dream pass** at 03:00 local via launchd: backlink refresh + conflict sweep + decay + session arc distillation (see Operating model section)
- `/biss-brain:remember <text>` for explicit user-curated memory (kind=decision auto-pins)
- UserPromptSubmit opt-in injection (off by default; see `[retrieve]` in config):
  - It skips slash commands, `!` prompts, acknowledgements and prompts under 3 words without loading the store (~70 ms).
  - Otherwise a memory is injected when its cosine is ≥ 0.6 or its BM25 is ≥ 7, top 1, within 6,000 chars.
  - A per-session ledger means nothing is injected twice, and anything already in core is skipped.
  - The embed budget is 300 ms; past it, the BM25 gate decides. The hard deadline is 2.5 s.
  - Each run appends latency and outcome to `inject_metrics.jsonl`; doctor warns when p95 exceeds 800 ms.
- Local web UI via `/biss-brain:ui`: next-session preview, search, review queue, session replay, timeline and status (see the Web UI section of the README)

## Pitfalls and current limitations

- **`claude -p` recursion.** Distillation calls spawn child Claude Code processes. They run with `--setting-sources ""`, `--strict-mcp-config` and no tools, so no hooks or plugins load in the child (not `--bare`, which ignores a keychain login). If you change the plugin's hook scope, double-check the child still runs cleanly.
- **Concurrent sub-agents.** A subagent's own transcript is distilled as its own segment, but tool output it returns to the parent is part of the parent's segment too.
- **Conflict judging is batched.** flush judges the candidate pairs of a segment in one call after storing it; the nightly dream sweep retries deferred and failed pairs. A contradiction can sit next to the older fact until one of those runs.
- **First session is silent.** No memory exists until the first `Stop` distillation lands — expect the brain to feel useful only after a few sessions on a project.
- **Tuning is partly unvalidated.** Half-life and debounce windows are still guesses.
  - The injection gate was fit on the v0.7 eval (`scripts/eval/calibrate_inject.py`).
  - Precision is 0.75–0.81 in-sample and 0.60–0.68 cross-suite, from 16–26 injected items per suite.
- **Prompt injection is slow on the full path.** Each prompt starts a new process, and importing lancedb alone takes ~0.8 s.
  - Wall clock on a 5.8 MB store: p50 1.3 s, p95 1.7 s.
  - That is why it stays off by default. A resident query process would remove the cost.

## Upgrading from v0.6 to v0.7

v0.7 changes how memories are captured, distilled and tagged (provenance, event time, one JSON
distill per session segment). Memories written by v0.6 have none of that, so the recommended
path is a fresh start: the old brain is archived, never deleted, and config plus core memory are
carried over.

1. Choose the embed model now (`[embed]` in `config.toml`). The new store records it in
   `store.json`, and every reader refuses a store built with a different model.
2. Update the plugin to v0.7, but don't run `/biss-brain:install` yet: it reloads the launchd
   jobs, and the DMN job starts at load. Create the new copy's venv instead (it touches only the
   plugin directory): `cd ~/.claude/plugins/cache/<marketplace>/biss-brain/0.7.0 && uv sync`.
3. Move the pinned memories you want to keep into core (`/biss-brain:remember --core`); they are
   not carried. Then quit every Claude Code session: their MCP servers count as running.
4. From a plain terminal, stop the background jobs:
   `launchctl bootout gui/$(id -u)/ch.bissbert.biss-brain.dream` (and `.dmn` if installed).
5. In the same terminal and plugin directory, dry run, then apply:
   ```bash
   scripts/_run.sh scripts/fresh_start.py           # what moves, what is carried
   scripts/_run.sh scripts/fresh_start.py --apply   # ~/.claude/brain → ~/.claude/brain-archive/brain-<ts>/
   ```
   It refuses while the flush daemon or any brain process runs. If it prints only
   "venv missing", step 2's `uv sync` did not run.
6. `/biss-brain:doctor`, then reload the jobs with `/biss-brain:install` (it rewrites the plists
   for the new plugin path) or `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<label>.plist`.

To keep the old memories searchable instead, `scripts/run_v07_migrations.sh` migrates the store
in place after writing a tar to `~/.claude/brain-backups/`. A migrated store has no embed
fingerprint, and the doctor says so. Behaviour changes are listed in [CHANGELOG.md](../CHANGELOG.md).

## Roadmap

- [x] Auto-invoke conflict judge — moved into the nightly dream pass (Stage 1b) rather than the hot insert path so it doesn't slow flushes
- [x] Distill sub-agent transcripts as their own segments (v0.7)
- [ ] `kind=note` flag on `/biss-brain:remember` (currently only `kind=decision` is special-cased)
- [ ] MCP server auto-registration smoke test
- [ ] Tune half-life, debounce, and inject threshold against real session data
- [ ] **Stage 3 — semantic clusters and community reports** (see below)

### Stage 3: semantic clusters and community reports (future)

Stages 1 and 2 cover within-session consolidation. Stage 3 is the cross-session layer: cluster the project's full embedding space into themes and write a "community report" per cluster. Modelled after Microsoft's GraphRAG community summarisation.

Sketch of what an implementation would look like:

- **New dependency.** `scikit-learn>=1.4` (HDBSCAN). ~30 MB; opt-in via `[clusters] enabled = true`.
- **New launchd job** `ch.bissbert.biss-brain.weekly`, written by `install.py` like the dream job. Runs Sundays at 04:00 local time. Cluster recomputation is too expensive to run nightly.
- **Schema changes.**
  - `chunks.cluster_id TEXT NULL` — points at the cluster a chunk belongs to.
  - New `clusters` SQLite table:
    ```sql
    CREATE TABLE clusters (
      id              TEXT PRIMARY KEY,        -- uuid
      project         TEXT NOT NULL,
      label           TEXT NOT NULL,           -- LLM-generated short name
      summary         TEXT NOT NULL,           -- "community report" body
      member_count    INTEGER NOT NULL,
      centroid_blob   BLOB,                    -- pickled np.ndarray for stable IDs across runs
      last_built_ts   INTEGER NOT NULL
    );
    ```
- **Algorithm.**
  1. Pull every active chunk's vector for the project (LanceDB scan).
  2. `hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')` over the matrix.
  3. For clusters with `≥ N` members, distill via `claude -p`: pass member texts (plus their arc parents from Stage 2) and ask for `{label, summary}` JSON. New prompt at `prompts/dream_cluster.md`.
  4. Match new clusters to the previous run's centroids by nearest-neighbour to keep `cluster_id` stable; emit a new id only when no centroid is within `0.3` cosine distance.
  5. Write the `clusters` row, set `chunks.cluster_id` for every member.
- **UI.** Search grows a cluster filter chip, and cards name their cluster. New `/biss-brain:clusters` command lists clusters by size with their labels.
- **Retrieval.** At search time, after fetching top-K results, also fetch the cluster summary for each result's `cluster_id` and offer it as an optional context block — gives the model the wider story before it dives into specifics.
- **Cost.** Rough estimate for a 2000-chunk project: ~30–60 clusters → ~30 Sonnet calls per weekly run. Acceptable for a Pro subscription.

The schema additions are **forward-compatible** with the current Stages 1+2 — adding `cluster_id` is one `ALTER TABLE` and the `clusters` table is new. No migration of existing data is needed.
