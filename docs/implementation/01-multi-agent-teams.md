# Multi-Agent Implementation Plan

> The headline deliverable. Maps every work stream in `00-roadmap.md` to a
> specific agent type, builder/reviewer team, and coordination protocol.
> Designed so each phase can be dispatched to parallel agents from a single
> Claude Code session, with reviewer agents auditing in series before the
> feature flag flips.

---

## 1. Why Multi-Agent

The work in v0.5 alone touches `flush.py`, `recall.py`, `inject_context.py`,
`dream.py`, `mcp/server.py`, `common.py`, `decay.py`, plus a new `dmn.py` and
the `_global` directory plumbing — six independent code paths that share a
schema migration but otherwise don't depend on each other. Done sequentially
by one agent, this is a several-day context window. Done in parallel by six
specialised agents in worktrees, it is hours.

The risk pattern is also non-uniform:

- **Capture-side changes** (salience, reconsolidation) are pure backend
  Python.
- **Cross-project paths** are security-critical (R2 silent supersession).
- **REM** is the highest-risk new component (self-reinforcing false belief).

These call for different reviewer mixes. A single generalist agent applies a
single review lens; multi-agent lets us route REM to a dedicated security
review, schema work to a dedicated documentation pass, and so on.

---

## 2. Agent Types Used

The plan uses agents already available in this user's environment. Each
phase's team table names them explicitly. Capabilities recap:

| Agent type | Strength | Used for |
|---|---|---|
| `Plan` | Architecture planning, no edits | Phase coordinator: produces step-by-step implementation plans, identifies critical files, calls trade-offs |
| `Explore` | Read-only fast search | Pre-phase discovery; locating call sites; cross-file grep |
| `biss-python:python-backend-engineer` | FastAPI, SQLAlchemy, services | Application code: dream stages, MCP server changes, daemons |
| `biss-python:python-expert-developer` | Language patterns, async, types | Hot-path code: typed-edge schema migration, sparse-sig helper, Hopfield blend |
| `biss-python:python-project-assistant` | Tooling, CI, packaging, daemons | Idle-detection daemon scaffolding, CI gates, version bumps |
| `biss-typescript:project-assistant` | TypeScript / web scaffold | UI route additions for hypothesis panel + schema viewer |
| `biss-design-system:design-assistant` | Design tokens, styled components | Distinct visual treatment for hypothesis chunks; schema graph viewer styling |
| `system-security-reviewer` | Read-only OWASP/STRIDE audit | Cross-project paths (R2/R3/R4); REM quarantine verification |
| `documentation-architect` | Read codebase → docs | Architecture docs after each phase; updates to docs/ |
| `general-purpose` | Catch-all multi-step | Orchestration glue when no specialist fits |

Reviewer agents (security, documentation) are **read-only** by convention —
they produce findings, not commits. Builders apply the findings.

---

## 3. Team Topology

Each phase has exactly three roles:

**Coordinator** (1 agent). Owns the phase. Writes the per-phase plan via the
`Plan` agent, dispatches builders, gathers their PRs, runs reviewers, flips
the feature flag. The coordinator is the **only** agent that talks to the
human.

**Builder teams** (1–6 agents per phase, parallel). Each team owns one work
stream (one or two files plus tests). Builders work in **isolated git
worktrees** so their edits never collide. Coordinator merges sequentially
after each builder reports.

**Reviewer team** (1–3 agents per phase, serial after builders). Quality,
security, documentation — each reads the merged result, files findings, and
either signs off or sends one builder back to fix.

```
                        ┌────────────────────┐
                        │    Coordinator     │
                        │  (Plan agent)      │
                        └──────┬─────────────┘
                               │ dispatch
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
      ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
      │ Builder team │  │ Builder team │... │ Builder team │
      │   (parallel) │  │   (parallel) │    │   (parallel) │
      └──────┬───────┘  └──────┬───────┘    └──────┬───────┘
             │                 │                   │
             └────── merge to phase branch ────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Reviewers   │
                        │  (serial)    │
                        └──────┬───────┘
                               │
                          fix? │ pass
                       ◄───────┴───────►
                                       │
                                       ▼
                                 flag-on / ship
```

---

## 4. Per-Phase Team Assignments

### v0.5 — Hebbian Foundation

**Coordinator.** `Plan` agent. Produces the v0.5 work-breakdown structure
referencing `02-v0.5-spec.md`. Owns the phase branch.

**Builder teams (parallel, 6 streams):**

| Team | Agent type | Files | Deliverable |
|---|---|---|---|
| **Capture** | `biss-python:python-backend-engineer` | `scripts/flush.py`, `scripts/common.py` | Salience scoring (heuristic + LLM-opt-in); schema migration: `salience`, `labile_until`, `reconsolidation_count`, `co_retrieval_count`, `cross_links`, `cross_eligible` |
| **Edges** | `biss-python:python-expert-developer` | `scripts/common.py`, `scripts/backlinks.py`, `scripts/dream.py` (NREM-1 only) | Typed `related_ids` schema + read-side adapter for both shapes; STDP `edge_bumps` SQLite table; NREM-2 drainer |
| **Recall** | `biss-python:python-backend-engineer` | `scripts/recall.py`, `scripts/inject_context.py`, `mcp/server.py` | STDP enqueue on co-retrieval; RIF apply; labile-window stamping |
| **Reconsolidation** | `biss-python:python-expert-developer` | `scripts/flush.py` | Pre-insert labile check; LLM merge call; counter logic; lesson promotion at N=5 |
| **DMN** | `biss-python:python-project-assistant` | `scripts/dmn.py` (NEW), packaging | Idle-detection daemon; loops over recent project; STDP drain + backlink refresh; no-LLM path |
| **Global** | `biss-python:python-backend-engineer` | `scripts/dream.py` (`_stage2a` + `_stage2b`), `scripts/recall.py`, `mcp/server.py`, `scripts/common.py`, `prompts/judge_conflict.md` | `_global` project dir; cross-project promotion gates; conflict-judge prompt with project labels; cross_links column writes |

Builders work concurrently in worktrees:

```
Agent({
  description: "v0.5 Capture team",
  subagent_type: "biss-python:python-backend-engineer",
  isolation: "worktree",
  prompt: "<v0.5 spec link, scope, tests required, definition of done>"
})
```

(Six such Agent calls dispatched in a single message — see §6 dispatch templates.)

**Reviewer team (serial, after merge):**

1. **Quality** — `biss-python:python-expert-developer`. Reads the merged
   diff, checks: (a) tests cover the migration path for both old and new
   `related_ids` shapes, (b) STDP queue does not race with concurrent
   recalls, (c) reconsolidation merge prompt preserves facts (golden-set
   spot-check).

2. **Security** — `system-security-reviewer`. Specifically audits the cross-
   project code path against `docs/research/04-cross-project-risk-audit.md`
   R1–R4. Produces a written sign-off referencing each risk by ID.
   **Blocking.**

3. **Documentation** — `documentation-architect`. Updates
   `docs/architecture/00-overview.md` with anything that drifted; verifies
   `docs/implementation/02-v0.5-spec.md` matches what landed.

**Risk routing.** The Global team's output goes to security review FIRST
before any flag flip. Cross-project supersession verification (R2) is a
codebase grep audit performed by the reviewer: every retrieval path that
sets `superseded_by` must be inspected.

**Estimated wall time.** ~6 hours of parallel builder work, ~2 hours
reviewer.

---

### v0.6 — Pattern Separation

**Coordinator.** `Plan` agent.

**Builder teams (parallel, 3 streams):**

| Team | Agent type | Files | Deliverable |
|---|---|---|---|
| **Sparse** | `biss-python:python-expert-developer` | `scripts/common.py` | `sparse_sig` BLOB column; top-k binarisation helper; sparse-Jaccard helper; pack/unpack of 4096-bit bitmap as 512-byte BLOB |
| **Index** | `biss-python:python-backend-engineer` | `scripts/flush.py`, `scripts/recall.py`, `scripts/inject_context.py`, `mcp/server.py`, `scripts/dmn.py` | Flush-time signature compute; retrieval pipeline insert sparse-Jaccard filter; DMN backfill stage with resumable batch state |
| **Reconsolidation-Merge** | `biss-python:python-expert-developer` | `scripts/flush.py` (extending v0.5 reconsolidation) | Add sparse-Jaccard ≥ 0.20 gate to the merge condition; tests showing surface-similar-different-domain chunks no longer merge |

**Reviewer team:**

1. **Quality** — `biss-python:python-expert-developer`. Verifies the
   bit-packing is correct (round-trip test); benchmarks sparse-Jaccard at
   k=20 to confirm < 1ms. Reviews the backfill resume protocol.

2. **Documentation** — `documentation-architect`.

**Estimated wall time.** ~4 hours parallel build, ~1 hour review.

---

### v0.7 — Semantic Scaffold

**Coordinator.** `Plan` agent.

**Builder teams (parallel, 4 streams):**

| Team | Agent type | Files | Deliverable |
|---|---|---|---|
| **Schema** | `biss-python:python-backend-engineer` | `scripts/schema_extract.py` (NEW), `scripts/common.py`, `prompts/extract_schema.md` (NEW) | `schema_edges` table; LLM extraction; bi-temporal versioning (`invalidated_ts`); CRUD helpers |
| **Fast-Path** | `biss-python:python-expert-developer` | `scripts/dream.py` (NREM-3 sub-stage) | Schema-consistency check; arcs that match all-existing-edges write `consolidation_round=2` directly |
| **REM** | `biss-python:python-backend-engineer` | `scripts/dream.py` (`_stage3_rem` NEW), `prompts/rem_recombine.md` (NEW) | High-salience low-similarity sampling; LLM recombination; `kind=hypothesis` insert with quarantine flag; verification that no auto-recall path includes `kind='hypothesis'` |
| **UI** | `biss-design-system:design-assistant` + `biss-typescript:project-assistant` | `ui/` templates and routes | Schema graph viewer route; hypothesis panel with promote/discard affordance; distinct visual treatment for hypotheses |

**Reviewer team:**

1. **Quality** — `biss-python:python-expert-developer`. Schema extraction
   precision audit on a 50-arc hand-labelled set; fast-path correctness.

2. **Security (REM-focused)** — `system-security-reviewer`. **Highest-risk
   review of the project.** Verifies via codebase grep that
   `kind = 'hypothesis'` is excluded from every retrieval path:
   `recall.py`, `inject_context.py`, `mcp/server.py`. Confirms that
   `_stage3_rem` writes `cross_eligible=0` so REM hypotheses can never
   auto-promote to L4. Produces a written audit listing every retrieval path
   with the kind-filter inspected. **Blocking.**

3. **Documentation** — `documentation-architect`. Updates the architecture
   doc; adds a "REM safety contract" section to docs/.

**Risk routing.** REM landing is gated on the security audit. The feature
flag `dream.rem.enabled` defaults to `false` for one full release after
v0.7 ships, only flipping on after observed UI promotion behaviour
confirms hypotheses are useful and never leak into auto-recall.

**Estimated wall time.** ~10 hours parallel build, ~3 hours review.

---

### v0.8 — Forgetting & Soft Recall

**Coordinator.** `Plan` agent.

**Builder teams (parallel, 3 streams):**

| Team | Agent type | Files | Deliverable |
|---|---|---|---|
| **Forget** | `biss-python:python-backend-engineer` | `scripts/decay.py`, `scripts/dream.py` (NREM-3 prune), archival flow | TTL prune logic; soft-delete via `superseded_by='__decayed__'`; cold archive writer; restore tooling |
| **Hopfield** | `biss-python:python-expert-developer` | `scripts/recall.py`, `scripts/inject_context.py`, `mcp/server.py`, `scripts/common.py` | Modern Hopfield blend (Ramsauer 2020): one matrix-multiply + softmax over candidate matrix; temperature config |
| **Iterative** | `biss-python:python-expert-developer` | `scripts/recall.py` | 3-iteration settle on the Hopfield-blended state; convergence threshold |

**Reviewer team:**

1. **Quality** — `biss-python:python-expert-developer`. Recall@10 regression
   suite must pass; iterative settle latency budget; archive round-trip test.

2. **Documentation** — `documentation-architect`.

**Estimated wall time.** ~6 hours parallel build, ~2 hours review.

---

### v1.0 — Trained Sparse Features (Research)

**Coordinator.** `Plan` agent. **Note:** v1.0 is research and may not ship
as default. The plan should produce a go/no-go recommendation, not a
guaranteed merge.

**Builder teams (serial, 2 streams — they share the SAE training output):**

| Team | Agent type | Files | Deliverable |
|---|---|---|---|
| **SAE-train** | `biss-python:python-expert-developer` | `scripts/sae_train.py` (NEW) | Per-project SAE trainer (PyTorch); feature persistence; resumable checkpoints |
| **SAE-integrate** | `biss-python:python-backend-engineer` | `scripts/common.py`, `scripts/flush.py`, `scripts/dream.py` (HippoRAG-style PPR) | Replace top-k binarisation with SAE activations; PPR over `schema_edges` for graph expansion |

**Reviewer team:**

1. **Quality** — `biss-python:python-expert-developer`. Recall@10
   improvement vs v0.8 must be measurable. Train cost amortisation test.

2. **Documentation** — `documentation-architect`.

**Decision gate.** If recall lift is not measurable, ship as opt-in
(`retrieve.use_sae = false` default), do not flip. Document the negative
result.

**Estimated wall time.** Months. Research timeline.

---

## 5. Coordination Protocol

### 5.1 Worktree isolation

Every builder agent is dispatched with `isolation: "worktree"`. This creates
a temporary git worktree, the agent works inside it, and merges happen at
the coordinator level. Two builders editing the same file (which should be
rare given the team boundaries above) merge with explicit human-in-the-loop
review at the coordinator.

### 5.2 PR-style change sets

Each builder produces:

- A focused diff (only the files in their team scope).
- A test set covering the new behaviour.
- A short report (under 300 words) summarising what changed and any decisions
  taken.

The coordinator collects these, merges them onto the phase branch, runs the
combined test suite, and only then dispatches reviewers.

### 5.3 Reviewer findings format

Reviewers (security, quality, documentation) produce findings in the
biss-reportcraft universal finding schema (per the plugin workspace's `CLAUDE.md`):

```yaml
finding:
  id: "F-v05-S-001"
  severity: "critical|high|medium|low|info"
  category: "security|quality|documentation"
  description: "..."
  location: "scripts/dream.py:412"
  recommendation: "..."
```

A `critical` or `high` finding blocks flag-flip. A `medium` finding requires
a fix or an explicit waiver from the user. `low` and `info` go into a
follow-up backlog.

### 5.4 Documentation always merges last

The documentation reviewer agent runs **after** code reviewers have signed
off and any required fixes have landed. Architectural docs, the v0.5/v0.6/etc
spec docs, and `99-cross-cutting.md` get updated in one final pass before
flag-flip. This prevents mid-phase doc churn.

### 5.5 The human's role

The coordinator agent is the only agent that surfaces to the human. The
human:

- Approves the phase plan before builders dispatch.
- Reviews any reviewer-blocking finding the coordinator surfaces.
- Authorises feature-flag flips at end of phase.

Builders and reviewers do not interact with the human directly.

---

## 6. Dispatch Templates

### 6.1 Coordinator kick-off

```
Agent({
  description: "v0.5 phase coordinator",
  subagent_type: "Plan",
  prompt: "Coordinate v0.5 of biss-brain (Hebbian Foundation phase).
  Read docs/implementation/00-roadmap.md §v0.5 and
  docs/implementation/02-v0.5-spec.md. Produce a step-by-step plan that
  identifies the six builder teams (Capture, Edges, Recall, Reconsolidation,
  DMN, Global), their files, dependencies, and the reviewer sequence.
  Identify any cross-team coordination needed for the schema migration
  (which must land before any read/write path can be tested). Return:
  (1) the dispatch order, (2) any flagged risk that needs human approval
  before builders fire, (3) the exit criteria for the phase."
})
```

### 6.2 Builder dispatch (six in one message)

```
[Agent#1: Capture team — biss-python:python-backend-engineer, isolation: worktree]
[Agent#2: Edges team — biss-python:python-expert-developer, isolation: worktree]
[Agent#3: Recall team — biss-python:python-backend-engineer, isolation: worktree]
[Agent#4: Reconsolidation team — biss-python:python-expert-developer, isolation: worktree]
[Agent#5: DMN team — biss-python:python-project-assistant, isolation: worktree]
[Agent#6: Global team — biss-python:python-backend-engineer, isolation: worktree]
```

Each prompt is self-contained:

```
"Implement the <Capture | Edges | Recall | ...> stream of biss-brain v0.5.

Read docs/implementation/02-v0.5-spec.md §<Stream Name> for the spec.
Scope: <files listed>. Out of scope: <files listed>.

Required tests: <list>. The schema migration in scripts/common.py is
authoritative — coordinate via the migration version number.

Definition of done:
- All listed files modified or created.
- Tests pass locally.
- Short report under 300 words: what changed, what decisions you took,
  any spec ambiguity you flagged.

Do not modify files outside your scope. If your stream depends on another
stream's output (e.g., the typed `related_ids` shape from Edges), assume
the migrated shape and tolerate the legacy shape via the read adapter."
```

### 6.3 Reviewer dispatch

Quality:

```
Agent({
  description: "v0.5 quality review",
  subagent_type: "biss-python:python-expert-developer",
  prompt: "Audit the merged v0.5 phase branch of biss-brain.
  Specifically: (a) tests cover both legacy and typed `related_ids` shapes,
  (b) STDP edge_bumps queue is not racy under concurrent recalls,
  (c) reconsolidation merge prompt preserves facts on a 5-chunk golden set.
  Produce findings in the universal finding schema. Mark critical/high if
  any of (a)/(b)/(c) fails."
})
```

Security:

```
Agent({
  description: "v0.5 cross-project security review",
  subagent_type: "system-security-reviewer",
  prompt: "Audit the cross-project code path in biss-brain v0.5 against
  docs/research/04-cross-project-risk-audit.md risks R1-R4. Verify by
  codebase grep that:
  (R1) hash dedup is per-project — search for the dedup hash key
       construction in scripts/flush.py and scripts/remember.py.
  (R2) cross-project supersession is permanently forbidden — search for
       every call site that writes `superseded_by`. Each must have a
       same-project guard in the WHERE.
  (R3) retrieval_count is per-project, not global.
  (R4) cross-project backlinks write to `cross_links`, not `related_ids`,
       and require cosine ≥ 0.85.
  Produce findings in the universal finding schema. Critical findings
  block flag-flip."
})
```

Documentation:

```
Agent({
  description: "v0.5 documentation review",
  subagent_type: "documentation-architect",
  prompt: "Audit and update biss-brain docs to reflect what landed in v0.5.
  Update docs/architecture/00-overview.md, docs/implementation/02-v0.5-spec.md,
  and docs/implementation/99-cross-cutting.md. Verify the schema migration
  table, config-key index, and feature-flag list are current."
})
```

---

## 7. Risk Mitigation Matrix

| Risk | Phase | Mitigation team | Contract |
|---|---|---|---|
| R1 Hash dedup collapse cross-project | v0.5 | Security reviewer | Audit dedup key construction; per-project scope mandatory |
| R2 Conflict judge silent supersession | v0.5 | Security reviewer | Every `superseded_by` write must have a same-project guard in WHERE |
| R3 Global retrieval_count bias | v0.5 | Security reviewer | Per-project retrieval counters; cross-project hits do not bump local counter |
| R4 Backlink false positives cross-project | v0.5 | Security reviewer | `cross_links` column distinct from `related_ids`; cosine ≥ 0.85 |
| REM self-reinforcing false belief | v0.7 | Security reviewer (dedicated) | Quarantine: every retrieval path excludes `kind='hypothesis'`; no auto-promotion to L4 |
| Sparse signature dimensionality wrong | v0.6 | Quality reviewer | Empirical capacity test on real corpus before flag-on |
| Hopfield blend brittleness | v0.8 | Quality reviewer | Recall@10 regression test on held-out queries |
| Pruning over-aggressive | v0.8 | Quality reviewer | Restore tooling proven before hard delete; 30-day soft-delete window |

---

## 8. Out-of-Band: Schema Migration Coordination

The single piece of state that crosses team boundaries within a phase is the
**schema migration**. v0.5 introduces six new columns and one new table;
every team consumes them.

Protocol:

1. The Edges team or Capture team (whichever the coordinator dispatches
   first) lands the migration in `scripts/common.py` first, in a single
   commit, on the phase branch directly.
2. All other teams branch off **after** that commit.
3. The migration is a pure additive `ALTER TABLE` — no team can change it
   later in the phase. If a team needs a column that isn't there, it
   surfaces to the coordinator, and the coordinator cuts a follow-up
   migration commit before any other builder pulls.

This eliminates merge conflicts on `common.py` and means the read adapter
for typed `related_ids` (the only non-additive change) is owned by exactly
one team (Edges).

---

## 9. Failure Modes

**A builder's diff doesn't pass tests.** Coordinator sends it back with the
test output. Builder agents are stateless across dispatches, so the prompt
must include the diff, the failing test, and the expected behaviour.

**A reviewer flags a critical finding.** Coordinator routes to the relevant
builder team (or, if it's architectural, surfaces to the human).

**Two builders edit the same file.** This should not happen given §4 team
boundaries. If it does, coordinator merges manually with human review.

**Phase exit criterion fails.** The flag does not flip. The phase is
"landed but not shipped" — the code is on `main` but disabled. Next phase
can still proceed if it does not depend on the failed criterion.

**The user wants to skip a phase.** Phases v0.5 → v0.8 are technically
independent (any can be flag-off) but later phases assume their schema
migrations have landed. Skipping a phase means flag-off, not no-code.

---

## 10. Phase-Zero Checklist (Before v0.5 Dispatch)

The coordinator must confirm before firing any builder:

- [ ] `docs/implementation/02-v0.5-spec.md` exists and lists every config
      key, schema column, and file under one section per team.
- [ ] `docs/implementation/99-cross-cutting.md` schema migration register
      has the v0.5 row with the version number and column list.
- [ ] The phase branch exists (e.g. `phase/v0.5-hebbian`) and is clean.
- [ ] The user has authorised the phase plan.
- [ ] The reviewer agents have access to `docs/research/04-cross-project-risk-audit.md`
      (security) and `docs/architecture/00-overview.md` (documentation).
- [ ] The cross-project feature flag default-off invariant is recorded in
      `99-cross-cutting.md` and verifiable by code grep.

---

## 11. Why This Plan Doesn't Use Two Other Tempting Patterns

**"One mega-agent that does everything."** Tempting, but the context window
required to hold the v0.5 spec + read every file + write every file +
review every file is far past the practical limit. Multi-agent splits the
context demand.

**"Pair-programming pattern (one builder, one reviewer per work stream)."**
Reviewer-per-stream produces six redundant security reviews where the
risks are concentrated in one stream (Global). Better to have one
specialised reviewer per concern, run after merge.

---

## 12. Beyond v1.0

Once v1.0 is research-shipped, the system has the full L0–L4 stack and
the dream-cycle state machine. Future phases (v1.1+) are not roadmapped
here — they would respond to observed failure modes from v0.5–v1.0
production use, not to a-priori design ambition. The right place to
capture that is a `docs/operations/` directory once it exists.
