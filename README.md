<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img alt="biss-brain: long-term memory for Claude Code" src="docs/assets/banner-light.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-8250df"></a>
  <img alt="Platform: macOS" src="https://img.shields.io/badge/platform-macOS-555">
  <img alt="Claude Code plugin" src="https://img.shields.io/badge/Claude%20Code-plugin-0969da">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776ab">
</p>

biss-brain gives Claude Code a memory that lasts beyond one session. While you work, it
quietly notes the decisions, fixes and lessons from each conversation. When you come back to the
project, Claude starts with what matters already in context, and can search the rest.

Everything is stored on your machine. Embeddings run locally through [Ollama](https://ollama.com),
the index is [LanceDB](https://lancedb.com) plus SQLite, and the model calls (turning a
transcript into short notes) go through your own Claude Code login. No API key, no cloud service.

## How it works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
    <img alt="Hooks queue events, a flush daemon scrubs, distills and embeds them into a local store; recall brings them back into later sessions; a nightly dream pass consolidates the store." src="docs/assets/pipeline-light.svg" width="100%">
  </picture>
</p>

1. **Capture.** Claude Code hooks (`Stop`, `SubagentStop`, `PreCompact`, `SessionEnd`) write a
   tiny event and return at once, so your session never waits on the brain.
2. **Distill.** A background daemon reads each finished stretch of conversation, strips
   `<private>` blocks and secrets, and asks Claude (Haiku by default) for a few short, typed
   notes: decisions, lessons, schemas, tasks.
3. **Store.** Notes are embedded locally with `nomic-embed-text` and stored per project, with
   links between related notes and a judge that marks outdated facts as superseded.
4. **Recall.** New sessions start with your core memory and the latest decisions for the project.
   Claude can search more through an MCP tool, and so can you with `/biss-brain:recall`.
5. **Dream.** Every night at 03:00 a launchd job refreshes links, sweeps for contradictions,
   decays stale notes and condenses whole sessions into summaries. It only adds; it never deletes.

```mermaid
sequenceDiagram
    autonumber
    participant You
    participant CC as Claude Code
    participant B as biss-brain
    You->>CC: "Let's use SQLite WAL mode for the queue"
    CC-->>B: Stop hook (event queued)
    B->>B: distill → decision: "queue uses SQLite in WAL mode"
    Note over B: embed, link, store in ~/.claude/brain
    You->>CC: next week, new session
    B-->>CC: SessionStart: core memory + recent decisions
    CC->>B: search_memory("queue locking")
    B-->>CC: the WAL decision, with date and source
```

## Built like a memory, not a log

Most memory tools append notes to a file or a vector store and search them. That works for a
week. After that the store fills with near-duplicates, old facts sit next to the ones that replaced
them, and nothing tells the important from the incidental. Human memory solves these problems
with a handful of known mechanisms, and biss-brain borrows them:

| In the brain | In biss-brain | Default |
|---|---|---|
| **Encoding.** An experience is stored as its gist, not a recording. | A finished stretch of conversation is distilled into a few typed notes (decision, lesson, schema, task), each with its source and date. | on |
| **Salience.** Novel experiences are encoded more strongly. | Every note gets a salience score from 1 to 10, from how new it is next to recent notes and how much it says. You can filter search by it. | on |
| **Consolidation in sleep.** The hippocampus replays the day and the cortex keeps the pattern. | The nightly *dream* pass condenses each session into an arc (round 1) and, if you want it, each day into a summary of arcs (round 2). A new session starts from arcs before raw notes. | on; daily summary opt-in |
| **Reconsolidation.** A recalled memory becomes editable for a while, then settles again. | A recalled note stays *labile* for five minutes. If what you say next revises it, the note is updated in place, not stored again beside itself. A note revised five times is promoted to a lesson. | on |
| **The forgetting curve.** Unused memories fade but are rarely erased. | Strength falls exponentially with age, pinned notes are kept five times stronger, and faint notes rank lower. Nothing is deleted. The UI lists what is fading so you can pin or forget it. | on |
| **Hebbian learning.** Cells that fire together wire together. | Notes recalled together three times or more are linked, so recalling one surfaces the others through `related_to`. | on |
| **Default mode network.** The resting brain does its own upkeep. | The dream pass makes those links nightly. An idle-time daemon can also make them during the day, between your sessions, and refresh the neighbours of new notes, with no model calls. | opt-in (`--with-dmn`) |
| **Memory updating.** A new fact overrides the old one, and you still remember there was an old one. | A judge compares each new note with its nearest older ones. A contradicted fact is marked replaced and drops out of recall, and its card shows what replaced it. | on |
| **Source monitoring.** Remembering *where* you learnt something, not only what. | Every note carries its origin. Notes drawn from tool output or web pages rank lower, are never injected on their own, and a low-trust note that reads like an instruction is quarantined. | on |
| **A sense of self.** A small set of facts that are always present. | Core memory is injected into every session, and nothing joins it without your yes. | on |

In v0.7 each mechanism that could be measured was switched off in turn against a frozen question
suite ([docs/eval/v0.7-ablation.md](docs/eval/v0.7-ablation.md)). One failed: *retrieval-induced forgetting*, where recalling a memory suppresses its rivals, made
answers worse and slowed every read, so it is off.

## Requirements

- **macOS.** The nightly jobs use launchd. The hooks may work elsewhere, but only macOS is tested.
- **[uv](https://docs.astral.sh/uv/)** (`brew install uv`). It builds the plugin's Python
  environment, about 1 GB, since the reranker pulls in PyTorch.
- **[Ollama](https://ollama.com)** (`brew install ollama`), running, for local embeddings.
- **Claude Code**, with the `claude` CLI logged in. Distillation runs `claude -p` on your
  subscription.

## Install

In Claude Code:

```
/plugin marketplace add Bissbert/biss-brain
/plugin install biss-brain@bissbert
/biss-brain:install
```

`/biss-brain:install` builds the environment, pulls `nomic-embed-text`, writes
`~/.claude/brain/config.toml` and schedules the nightly job. Restart Claude Code afterwards so the
hooks and the MCP server load, then run `/biss-brain:doctor` to check every dependency.

Expect the brain to stay quiet for the first session or two on a project: nothing is recalled
until something has been distilled.

### Updating

`/plugin update biss-brain@bissbert`, or let Claude Code auto-update the marketplace. Each
version gets a fresh plugin folder; on the first session after an update the plugin rebuilds its
environment in the background (a few minutes, announced at session start) and the nightly job
follows the new version by itself.

## Privacy

- **Stored locally.** Memories live in `~/.claude/brain/`, one folder per project. Embedding,
  indexing and search all run on your machine.
- **Claude calls go through your login.** Distilling notes, judging conflicts and the nightly
  session summaries run `claude -p` with your Claude Code login, the same service your sessions
  already talk to. Nothing goes anywhere else.
- **Scrubbed first.** Secrets (API keys, access tokens, private keys, passwords in URLs, other
  high-entropy strings) are removed before anything is distilled or embedded. Wrap text in
  `<private>…</private>` to keep it out entirely.
- **Scoped.** Your home folder itself, `~/Downloads`, `~/Library`, `~/.claude` and `/tmp` are
  excluded by default. Drop a `.biss-brain-ignore` file in any project to switch it off there.
- **Local UI.** The web UI binds to `127.0.0.1` only and loads nothing from the internet; its
  scripts and fonts ship with the plugin.

## Commands

| Command | What it does |
|---|---|
| `/biss-brain:install` | Build the environment, pull the embedding model, write config, schedule the nightly job. `--with-dmn` adds idle-time maintenance. |
| `/biss-brain:doctor` | Check every dependency and the store; says what to fix. |
| `/biss-brain:recall <query>` | Search this project's memory. |
| `/biss-brain:remember <text>` | Save a memory yourself. `--core` makes it core memory, injected every session. |
| `/biss-brain:core` | Review and accept the core-memory proposals Claude and the dream pass make. |
| `/biss-brain:forget <id>` | Retire a memory so it stops being recalled. |
| `/biss-brain:inject on\|off\|status` | Per-prompt injection for this project (off by default). |
| `/biss-brain:stats` | Counts and health for this project. |
| `/biss-brain:ui` | Open the local web UI. |

### Hooks and tools in detail

| Trigger | Behaviour |
|---|---|
| `Stop` hook | Queues a small event (transcript offset) after every assistant turn. The session segment is distilled once, when it closes. |
| `SubagentStop` hook | Queues a per-sub-agent event; a subagent's own transcript is distilled 10 s after it stops. |
| `SessionEnd` hook | Closes the session's segment. |
| `PreCompact` hook | Closes the segment and writes a digest of the conversation tail, then returns at once. The segment is distilled in the background (Sonnet). |
| `SessionStart` hook | Injects core memory in full (see below), then the top-3 recent decisions for the project. Also fires after `/clear`. |
| `SessionStart` (compact) | Re-injects core memory and this session's pre-compaction summary. If the summary is not distilled yet, it injects the conversation-tail digest instead, and the summary follows once on the next prompt. |
| `UserPromptSubmit` hook | Off by default (`[retrieve] user_prompt_inject`). When on, it injects at most one memory that clears the calibrated gate, once per session, within a 2.5 s deadline. |
| `/biss-brain:recall <q>` | Manual hybrid search of the project brain. |
| `/biss-brain:remember <text>` | Explicit user-curated memory (kind=decision auto-pins). `--core [--shared\|--global]` writes core memory instead. |
| `/biss-brain:core` | Show, review, accept or reject core-memory proposals; `shared on\|off`; `export` an `@import` line. |
| `/biss-brain:inject` | Prompt injection for this project: `on`, `off`, or `status` (gate, latency p50/p95, outcomes). |
| `/biss-brain:ui` | Launch a local web UI to browse and inspect the brain (see below). |
| MCP tool `search_memory` | Model-initiated retrieval during work. |
| MCP tools `core_propose_add` / `core_propose_remove` | The model suggests core changes; nothing changes until `/biss-brain:core accept`. |

## Web UI

`/biss-brain:ui` starts a local FastAPI server (random free port, `127.0.0.1` only) and opens your browser. Re-running the command reuses the live server. `/biss-brain:ui --stop` shuts it down.

Press `?` on any page for the keyboard shortcuts and a glossary of the terms the UI uses.

| Page | What it is for |
|---|---|
| **Overview** | What the next session in this project starts with: the core notes and memories the session-start hook would inject, and the exact text, read without side effects. Below it, every project with its size and last activity. |
| **Search** | Hybrid search over the project, with filters for kind, date, session, salience and consolidation round. Each card can be pinned or forgotten (forget can be undone), and says what it replaced or what replaced it. |
| **Review** | The few things that want a decision: core-memory proposals to accept or reject, the faintest fading memories to pin or forget, and facts replaced in the last 30 days next to what replaced them. |
| **Sessions** | Recent sessions, and a replay of each one: its notes in order, its subagents, and what each note replaced. |
| **Timeline** | Captures per day by kind, and how strength fades over time. |
| **Status** | The flush daemon, the queue, the nightly dream and idle-time maintenance, and the most recent model-call failure, if any. |

The UI only reads, except for pin, forget, labelling a project with its path, and accepting or
rejecting core proposals, which go through the same code as the slash commands. The server runs
detached, survives the terminal closing, and logs to `~/.claude/brain/ui.log`.

## Core memory

A small curated set of notes injected in full at every session start, ahead of recalled memory.
It comes in three layers, injected in this order:

| Layer | File | Written by |
|---|---|---|
| shared | `<repo>/.claude/brain/CORE.md`, committed with the code | You, with `--core --shared` or by hand. Injected only after you opt in for that repo (`/biss-brain:core shared on`); a cloned repo's file is never injected on its own. |
| private (default) | `~/.claude/brain/core/<repo>.md` | You. |
| global | `~/.claude/brain/core/_global.md` | You, with `--core --global`. |

- The files are the source of truth. Bullets carry `<!-- m:<id> <date> <source> -->`. Hand edits
  win, and deleting a line retires the memory; `core.sqlite` keeps each item's versions.
- Live core is written by `/biss-brain:remember --core`, `/biss-brain:core accept` and hand
  edits. Both commands are user-invoked only (`disable-model-invocation`). The plugin's MCP tools
  and the dream pass can only add proposals, because core skips the taint filter that keeps tool
  and web content out of injection. A model that runs the scripts through Bash still goes
  through Claude Code's Bash permission prompt.
- `~/.claude/brain/core/` is a git repo, committed on every change under the identity
  `biss-brain`, without signing and without touching your git config. A hand edit holding a
  secret-shaped line is never staged, by that commit or a later one. `/biss-brain:doctor`
  points at it, and injection and `/biss-brain:core` scrub on read.
- At most `[core] max_items` (200) bullets are injected, project layers first. Any extra items
  are announced in the injected text, never dropped silently. Doctor warns at `warn_ratio` (80%).
- If Claude Code's built-in auto memory is also on, doctor says so. To keep only biss-brain
  core, set `"autoMemoryEnabled": false` in `~/.claude/settings.json`.

## Configuration

Edit `~/.claude/brain/config.toml`. Defaults are sensible. Common knobs:

```toml
[distill.models]
stop = "haiku"            # cheap; called many times per session
subagent_stop = "haiku"
pre_compact = "sonnet"    # one call per compaction; quality matters
conflict_judge = "haiku"

[debounce]
segment_idle_seconds = 600  # a session segment closes after 10 min without a Stop
segment_max_events = 20     # … or at 20 Stops, at SessionEnd, or at PreCompact
subagent_stop_seconds = 10

[scope]
exclude = ["~", "~/Downloads", "~/Library", "~/.claude", "/tmp"]  # "~" exactly, the rest as trees
include_only = []           # set it to capture only under these directories
```

To turn the brain off for one project, add a `.biss-brain-ignore` file, or
`.claude/brain.toml` with `enabled = false`, at its root (or any parent).
Excluded and opted-out directories stop in `_run.sh` before Python starts.

## Uninstall

Run the uninstall script from the installed plugin folder first, then remove the plugin:

```bash
cd ~/.claude/plugins/cache/bissbert/biss-brain/<version>
./uninstall.sh                # stops the jobs, removes the runner and the venv; keeps memories
./uninstall.sh --purge-data   # also deletes ~/.claude/brain (asks first)
```

```
/plugin uninstall biss-brain@bissbert
```

Without `--purge-data`, your memories stay in `~/.claude/brain/`, so a reinstall picks up where
you left off.

## Documentation

- [docs/internals.md](docs/internals.md): stack, storage layout, the flush daemon, the dream
  pass, which mechanisms are on and why, known limits, roadmap.
- [docs/README.md](docs/README.md): the research and design notes behind it.
- [CHANGELOG.md](CHANGELOG.md)

## License

[MIT](LICENSE) © Bissbert
