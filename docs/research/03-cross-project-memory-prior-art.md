# Cross-Project / Cross-Context Memory: Prior Art Survey

> Reference for the L4 (global semantic store) layer. Surveys how production and
> research-grade agent-memory systems handle the per-context vs cross-context
> question.

---

## System-by-System Survey

### Letta / MemGPT

**Isolation model.** Three-tier architecture: *core memory* (in-context blocks,
analogous to RAM), *recall memory* (full conversation history, searchable), and
*archival memory* (vector / graph external store). Each agent has its own core memory
blocks. Agents can optionally share blocks — explicitly — via the multi-agent API,
where a "sleep-time agent" or memory-management agent can write to any block it is
given access to. **There is no implicit sharing.**

**Promotion rule.** No automatic promotion from episodic to semantic. Generalisation
is delegated to the agent's own reasoning: it can call memory-write tools to
reformulate a pattern as a new core-memory fact. This is deliberate and
write-tool-mediated, not triggered by frequency or tagging.

**Retrieval behaviour.** Recall memory is searched by embedding similarity + recency;
archival memory is pure embedding retrieval. Neither crosses agent boundaries unless
the store is explicitly shared.

**Failure modes.** Multi-agent shared blocks have no documented conflict-resolution
protocol for concurrent writes. The architecture paper acknowledges that paging the
wrong memory wastes context tokens; archiving too aggressively creates "memory
blindness."

Sources: <https://www.letta.com/blog/agent-memory>, <https://docs.letta.com/concepts/memgpt/>

---

### mem0

**Isolation model.** Four-scope identifier system: `app_id`, `user_id`, `agent_id`,
`run_id`. These compose at query time — a search scoped to `user_id=alice` physically
cannot touch `user_id=bob`'s graph (when using FalkorDB, each user maps to a separate
graph: `mem0_alice`, `mem0_bob`). **Cross-project memory requires explicit opt-in by
passing the same scope identifiers.**

**Promotion rule.** No explicit episodic-to-semantic promotion mechanism. Memory types
(episodic, semantic, procedural) are treated as parallel storage categories, not a
promotion ladder. The 2026 state-of-agent-memory report notes "cross-session identity
resolution" (i.e., recognising the same user across contexts) remains an open,
unsolved problem.

**Retrieval behaviour.** Hybrid: vector retrieval for semantic similarity + graph
traversal for relational facts (Neo4j / FalkorDB). Retrieval is strictly scoped to
the caller's identifiers.

**Failure modes.** "Memory staleness at scale" — facts don't expire, they become
"confidently wrong" rather than merely outdated. No documented vocabulary collision
or cross-context bleed by design, but staleness is acknowledged.

Sources: <https://mem0.ai/blog/state-of-ai-agent-memory-2026>,
<https://www.falkordb.com/blog/graph-memory-llm-agents-mem0-falkordb/>

---

### A-MEM (Xu et al., NeurIPS 2025)

**Isolation model.** Single flat memory store per agent. No multi-user or
multi-project partitioning; the unit of isolation is the agent, not the project or
session.

**Promotion rule.** Zettelkasten-inspired: every new memory generates a structured
note (keywords, contextual description, tags). The system then performs embedding
similarity search over existing notes and establishes links where similarity exceeds
a threshold. **Links are not typed** in the published version — they are
similarity-based associations without semantic role labels (e.g., no "contradicts,"
"generalises," "caused-by"). Crucially, linking a new note can trigger retroactive
updates to the contextual representations of existing notes, so the network evolves
rather than just accretes. **High-connectivity notes effectively become semantic
anchors.**

**Retrieval behaviour.** Traversal of the memory network, not flat vector search.
The network structure biases retrieval toward well-connected (frequently linked)
memories.

**Failure modes.** The paper does not report on cross-context bleed (single-agent
scope). The undirected, untyped link model is a design risk: similarity-based linking
can pull in surface-similar but contextually unrelated memories ("embeddings are good
at determining when text 'looks like' each other, but terrible at knowing 'this is
the cause'" — per the broader survey literature).

Sources: <https://arxiv.org/abs/2502.12110>, <https://github.com/WujiangXu/A-mem>

---

### ChatGPT Memory / Claude Project Memory

**Isolation model (ChatGPT).** Two tiers since August 2025: *account-wide memory*
(preferences, facts the model decides to store across all chats) and *project-only
memory* (explicitly opt-in, scoped to a single project). With project-only mode
enabled, the model draws context only from within-project conversations and
explicitly does not carry anything into chats outside the project. Cross-project
bleed is blocked at the product level. Account-wide memories, by contrast, are
automatically generalised from any conversation and surface everywhere — **this is
the one case of automatic cross-context promotion, and it has drawn criticism**
(Simon Willison: "I really don't like ChatGPT's new memory dossier").

**Promotion rule.** Account-wide: automatic and opaque — the model decides
unilaterally which facts are worth storing. Project-only: no cross-boundary
promotion. No published frequency or kind-tag signals.

**Retrieval behaviour.** Implicit — surfaced by the model during generation, not
explicit vector retrieval.

**Failure modes.** Account-wide memory is described by users as "hidden context" —
unexpected facts from unrelated conversations influencing responses. No documented
vocabulary collision, but semantic bleed between domains (personal vs. work) is a
reported UX concern. Claude does not have a published project-memory model
equivalent to ChatGPT's project-only mode as of the knowledge cutoff.

Sources: <https://help.openai.com/en/articles/10169521-using-projects-in-chatgpt>,
<https://simonwillison.net/2025/May/21/chatgpt-new-memory/>

---

### Zep / Graphiti

**Isolation model.** Temporal knowledge graph with dual timelines: event time (when
a fact occurred) and ingestion time (when it was added). Entity resolution across
sessions is done via embedding similarity + full-text search over existing graph
nodes — the same entity appearing in two sessions gets merged into one node with
temporal edge annotations. **There is no hard per-session boundary at the graph
level**; sessions are differentiated by edge timestamps, not by graph partitioning.

**Promotion rule.** Implicit: repeated entity mentions accumulate edges on the same
node, which effectively raises that entity's "centrality." Conflicting facts trigger
edge invalidation — the older edge gets a `t_invalid` timestamp, and the newer fact
supersedes it. This is temporal versioning, not abstraction promotion.

**Retrieval behaviour.** Graph traversal + temporal filtering. Can synthesise across
sessions (this is the design goal — cross-session information synthesis is
explicitly cited as an enterprise use case). 18.5% accuracy improvement on
LongMemEval, which specifically tests cross-session recall.

**Failure modes.** The published paper evaluates only single-conversation scenarios;
cross-user isolation, multi-tenant access control, and entity-deduplication across
independent users are not addressed. The entity resolution mechanism (embedding
similarity) is a latent risk: two different users' mentions of a common entity
("the API server") could merge into a shared node if the graph is not partitioned
per-user. The GraphZep TypeScript port adds per-user session isolation as an
explicit feature, implying the base Graphiti does not enforce this.

Sources: <https://arxiv.org/abs/2501.13956>, <https://github.com/getzep/graphiti>,
<https://github.com/aexy-io/graphzep>

---

## Episodic → Semantic Promotion: Academic Framing

The standard framing (from multiple 2024–2026 surveys) is that episodic memory is
session/context-scoped and semantic memory is de-contextualised, generalised
knowledge. Concrete promotion proposals:

| Signal | System / Paper | Notes |
|---|---|---|
| **Frequency** | Generative Agents, survey lit | After N recurrences of the same pattern, consolidate. Threshold is hand-tuned. |
| **Importance score** | Generative Agents, HiMem | Multi-signal: recency (exponential decay) + embedding relevance + self-assessed integer salience. High scorers promoted in reflection cycles. |
| **Semantic divergence threshold** | GAM (arXiv 2604.12285) | Cosine distance between consecutive episode embeddings exceeds ε → triggers consolidation. Sparse, event-driven, not continuous. |
| **Temporal clustering** | TSM (arXiv 2601.07468) | Monthly slices; Gaussian Mixture Model clusters entities within a slice into Topic/Persona summaries. Fixed cadence. |
| **Session-boundary / buffer-overflow** | GAM | Session end, natural pause, or episodic buffer overflow triggers semantic consolidation phase. |
| **Explicit kind tagging** | Largely absent | No surveyed production system uses a typed "lesson / decision / fact" taxonomy to gate promotion. **This remains a research gap.** |

The Achilles' heel is uniform: **consolidation is not principled.** The
episodic→semantic boundary is set by heuristic thresholds, not learned policies, and
no production system publishes a systematic rule.

---

## Documented Failure Modes for Cross-Context Consolidation

1. **Summarisation drift / compression loss.** Each consolidation pass discards
   low-frequency details. A rare-but-critical instruction from session 1 survives the
   first pass but vanishes by the third. (Survey, arXiv 2603.07670)
2. **Self-reinforcing false belief.** An incorrect generalisation ("API X always
   errors with param Y") is promoted to semantic memory and never corrected because
   the agent stops collecting contradicting evidence. (arXiv 2603.07670)
3. **Vocabulary collision / surface similarity bleed.** Untyped embedding-based
   linking (as in A-MEM) conflates text that *looks similar* with text that is
   causally or contextually related. A coding memory from project A linking to a
   semantically similar but contextually unrelated memory from project B is a
   natural consequence.
4. **Silent eviction.** When the wrong episodic memory is evicted or overwritten
   during consolidation, no exception fires. Degradation is silent. (arXiv 2603.07670)
5. **Temporal granularity mismatch.** Fixed-interval consolidation (TSM's monthly
   slices) destroys temporal precision for bursty, irregular workloads.

Sources:

- Memory for Autonomous LLM Agents survey: <https://arxiv.org/html/2603.07670v1>
- GAM hierarchical memory: <https://arxiv.org/html/2604.12285>
- TSM temporal semantic memory: <https://arxiv.org/html/2601.07468v1>
- Episodic memory position paper: <https://arxiv.org/abs/2502.06975>

---

## Synthesis: Prevailing Wisdom on Cross-Context Consolidation

**Is shared semantic memory opt-in or automatic?**
In every production system surveyed (Letta, mem0, Zep, ChatGPT Projects),
cross-context sharing is either hard-isolated by default or requires explicit opt-in
(same scope identifiers, shared block grant, or project-only mode disabled). The only
automatic cross-context promotion in production is ChatGPT's account-wide memory,
which is the design decision that has attracted the most user criticism. Academic
systems (A-MEM, GAM, TSM) are single-agent and sidestep the question entirely.
**The consensus strongly favours opt-in.**

**What signals do systems use to decide generalisability?**
In rough order of prevalence: (1) frequency / recurrence across sessions, (2)
importance scoring (recency + relevance + self-assessed salience), (3) semantic
divergence threshold at session boundaries, (4) temporal clustering into fixed-cadence
slices. **Explicit kind tagging** ("this is a lesson," "this is a decision," "this is
a project-specific note") is conspicuously absent from all surveyed systems — it
appears only in practitioner prescriptions, not in shipped systems. **This is a
genuine gap and arguably the highest-leverage design lever for biss-brain's dream
pass**, because kind-tagged memories can be promoted with far less risk of bleed
than similarity-promoted ones.

**Design implication for biss-brain's L4 layer.** Consolidation across project
contexts should be gated on explicit kind metadata (e.g., only `kind=lesson` or
`kind=decision` or `kind=schema` chunks are candidates for cross-project promotion),
not on raw embedding similarity. Frequency across projects is a useful secondary
signal, but untyped similarity-based linking across project boundaries is the
primary failure mode to avoid. If we do build cross-project links, **typing them**
(`generalises`, `contradicts`, `refines`) substantially reduces retrieval ambiguity —
something A-MEM's untyped links do not do and which the surveyed literature
explicitly calls a gap.
