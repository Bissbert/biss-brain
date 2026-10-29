# biss-brain Documentation

This is the v2 design corpus for biss-brain — the brain-inspired memory
plugin. The work is organised so each document has one job and links
forward to the next.

For how the shipped plugin works today (stack, storage, daemon, dream pass,
pitfalls), read [internals.md](internals.md) first.

## What's here

```
docs/
├── README.md                                 ← you are here
├── internals.md                              ← how the shipped plugin works (start here)
├── assets/                                   ← README graphics
├── research/                                 ← the evidence base
│   ├── 01-neuroscience-of-memory.md
│   ├── 02-bio-inspired-ai-architectures.md
│   ├── 03-cross-project-memory-prior-art.md
│   ├── 04-cross-project-risk-audit.md
│   ├── 05-dream-pass-hook-mapping.md
│   ├── 06-extended-literature.md
│   └── verification-report.md
├── architecture/                             ← the synthesis
│   └── 00-overview.md
└── implementation/                           ← the plan
    ├── 00-roadmap.md
    ├── 01-multi-agent-teams.md
    ├── 02-v0.5-spec.md
    ├── 03-impl-research-retrieval.md
    ├── 04-impl-research-llm-stages.md
    ├── 05-impl-research-orchestration.md
    └── 99-cross-cutting.md
```

## Reading order

If you have **30 minutes** and want the gist:

1. `architecture/00-overview.md` — seven invariants, ASCII diagram, layer
   definitions, sleep cycle, retrieval pipeline.
2. `implementation/00-roadmap.md` — phases v0.5 → v1.0 and what each
   delivers.
3. `implementation/01-multi-agent-teams.md` — how the work gets done.

If you are **about to build v0.5**:

1. `architecture/00-overview.md` (skim §1–§4).
2. `research/05-dream-pass-hook-mapping.md` (where the changes plug in).
3. `research/04-cross-project-risk-audit.md` (the R1–R4 contracts).
4. `implementation/02-v0.5-spec.md` (your spec).
5. `implementation/05-impl-research-orchestration.md` (dream-cycle state machine, performance budget, migration sequencing).
6. `implementation/03-impl-research-retrieval.md` and `04-impl-research-llm-stages.md` (per-stream code patterns and prompts).
7. `implementation/01-multi-agent-teams.md` §4 (your team's scope).
8. `implementation/99-cross-cutting.md` (config keys, flags, glossary).

If you are **reviewing** a phase:

1. The phase's spec doc (`02-v0.5-spec.md`, etc.).
2. `implementation/01-multi-agent-teams.md` §7 (risk matrix you must
   audit against) and §6.3 (your reviewer dispatch template).
3. `implementation/99-cross-cutting.md` §6 (invariants to grep for).

If you want to **understand the science**:

1. `research/01-neuroscience-of-memory.md`.
2. `research/02-bio-inspired-ai-architectures.md` for the AI mapping.

If you want to **understand the cross-project design**:

1. `research/03-cross-project-memory-prior-art.md` (industry consensus).
2. `research/04-cross-project-risk-audit.md` (the contracts).
3. `architecture/00-overview.md` §6 (the layer).
4. `implementation/02-v0.5-spec.md` §3.6 (the v0.5 spec).

## Document one-liners

**research/01-neuroscience-of-memory.md.** Compressed reference of the
neuroscience that informs the design — CLS, hippocampal subfields,
NREM/REM, replay, schemas, reconsolidation, active forgetting, engrams,
DMN, STDP, sparse codes. Ends with seven implications for software.

**research/02-bio-inspired-ai-architectures.md.** Skeptical survey of
HTM, modern Hopfield, NTM/DNC, replay-based continual learning,
predictive coding, VSA, schema-based memory, SAEs, SNNs. Per-technique
verdict: steal, watch, or skip.

**research/03-cross-project-memory-prior-art.md.** What Letta, mem0,
A-MEM, ChatGPT, Zep/Graphiti do about cross-context memory. Industry
consensus is overwhelmingly opt-in.

**research/04-cross-project-risk-audit.md.** What would break if
biss-brain's existing dedup, conflict-judge, decay, and backlink logic
saw cross-project data. Hard risks R1–R4, soft risks S1–S3, mitigations
M1–M4. The R2 invariant — cross-project supersession is permanently
forbidden — is the most important rule in the entire corpus.

**research/05-dream-pass-hook-mapping.md.** Where in `dream.py`,
`recall.py`, `inject_context.py`, and `mcp/server.py` each new piece
plugs in. Tables of cross-cutting file changes per phase.

**research/06-extended-literature.md.** Targeted extension surfaced after
verification. Five gap areas: REM safety in agentic memory; cross-session
identity resolution; sleep-time compute (LightMem, MemoryOS,
MemoryAgentBench); SAEs over retrieval embeddings (CL-SR, k-SAE, Gemma
Scope); salience and prediction-error write-gating (HippoRAG PPR,
SynapticRAG, Barry & Gerstner three-factor plasticity).

**research/verification-report.md.** Output of the citation verifier on
docs 01–03. 31 verified, 5 flagged, 2 approximate. Fixes from this report
have been applied to the source documents; the report itself is preserved
as the audit trail.

**architecture/00-overview.md.** The synthesis. Seven invariants,
L0–L4 layer model, sleep cycle as a state machine, CA3 retrieval
pipeline, cross-project layer rules, what we deliberately do NOT
build, phased rollout summary.

**implementation/00-roadmap.md.** Phases v0.5 → v1.0 with goals,
scope, exit criteria, dependencies, rollback strategy, success metrics.

**implementation/01-multi-agent-teams.md.** How to actually do the
work — coordinator/builder/reviewer topology, agent-type routing,
per-phase team tables, dispatch templates, risk-mitigation matrix,
schema-migration coordination protocol. The headline doc.

**implementation/02-v0.5-spec.md.** The concrete spec for v0.5 — schema
migrations, config keys, per-stream specs (Capture, Edges, Recall,
Reconsolidation, DMN, Global), required tests, files touched, definition
of done.

**implementation/03-impl-research-retrieval.md.** How to actually code
the retrieval primitives — sparse-signature top-k binarisation
(`np.argpartition` + `np.packbits().view(uint64)`, 256-entry LUT for
Jaccard at 41 µs), modern Hopfield blend (β = 10–25 for normalised
cosines, 3-iter settle in 490 µs), STDP co-retrieval queue
(`BEGIN IMMEDIATE` drainer with `GROUP BY ... HAVING ... ON CONFLICT`),
reconsolidation compound index, RIF async on NREM-2.

**implementation/04-impl-research-llm-stages.md.** How to actually code
the LLM-driven stages — salience formula `0.4·length + 0.6·surprise`,
schema extraction with JSON output and three-temperature self-consistency,
reconsolidation cosine-drift gate (≤ 0.12, novel contribution), REM
HYPOTHESIS prefix with 0.30 grounding gate and manual UI promotion,
cross-project judge as a 4-way decision tree with structural caller-side
guard for the R2 invariant.

**implementation/05-impl-research-orchestration.md.** Operational glue
tying retrieval and LLM streams together — codebase-audit summary of
helpers to reuse (`load_config`, `project_key`, `hybrid_search`,
`judge_conflict`, `distill`), dream-cycle state machine with new
`dream_state` SQLite table for resumability, DMN daemon with macOS idle
detection (`ioreg`), cross-project promotion two-phase atomic write,
per-stage performance budget (recall hot path ≈ 125 ms vs 200 ms
budget), observability JSON-line logging, M0–M6 migration sequencing
within v0.5, failure-modes table, and three open empirical calibration
questions (salience α/β weights, Hopfield β, drift threshold 0.12).

**implementation/99-cross-cutting.md.** Schema migration register,
config-key index, feature-flag list, reserved sentinels, glossary,
code-path invariants reviewers grep for.

## Conventions

- **Filenames** are zero-padded numeric prefixes (`01-`, `02-`, ...) for
  reading order. New documents within a category should pick the next
  number.
- **Cross-references** use repo-relative paths
  (`docs/research/04-cross-project-risk-audit.md`).
- **Code references** use `path:line` (`scripts/dream.py:248`).
- **Findings** in reviews use the biss-reportcraft universal finding
  schema (see `01-multi-agent-teams.md` §5.3).

## Contributing changes

Updates to these docs flow through the documentation reviewer agent at
the end of each phase (see `01-multi-agent-teams.md` §5.4). Mid-phase
edits should stay localised to a phase's spec doc; architecture and
research docs are stable artefacts that change only when an invariant
changes.

A change that revises an invariant (the seven from `00-overview.md`,
the R1–R4 contracts, the default-off cross-project flag) is by
definition a breaking architectural change and requires a major-version
bump and explicit user sign-off.

## Status

| Document | Status |
|---|---|
| research/01-neuroscience-of-memory.md | Stable |
| research/02-bio-inspired-ai-architectures.md | Stable |
| research/03-cross-project-memory-prior-art.md | Stable |
| research/04-cross-project-risk-audit.md | Stable |
| research/05-dream-pass-hook-mapping.md | Stable |
| research/06-extended-literature.md | Stable |
| research/verification-report.md | Living — re-run on each citation update |
| architecture/00-overview.md | Stable |
| implementation/00-roadmap.md | Stable |
| implementation/01-multi-agent-teams.md | Stable |
| implementation/02-v0.5-spec.md | Shipped |
| implementation/03-impl-research-retrieval.md | Stable |
| implementation/04-impl-research-llm-stages.md | Stable |
| implementation/05-impl-research-orchestration.md | Stable |
| implementation/99-cross-cutting.md | Living — updated each phase |

"Stable" means the document captures the current design intent. It will
be updated by the documentation reviewer when a phase ships.
"Ready to build" means the spec is complete enough that a builder team
can be dispatched against it.
