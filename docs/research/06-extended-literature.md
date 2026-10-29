# Extended Literature: Five Gap Areas for biss-brain v2

> Targeted search extending the research corpus (docs 01–05). Every citation is drawn
> from a live API call to OpenAlex, Crossref, arXiv, or Semantic Scholar. Sources
> without a resolvable DOI or arXiv ID are quarantined. Confidence tiers follow the
> taxonomy in `skills/anti-hallucination/references/confidence-taxonomy.md`.

Search date: 2026-04-30. APIs: OpenAlex (primary), Crossref, arXiv, Semantic Scholar
(rate-limited; supplementary). PRISMA inclusions: 17 verified sources across five gap
areas; 1 quarantined (SCM — no persistent DOI). 47 records identified; 21 screened;
19 assessed for abstract availability; 18 included.

---

## Gap 1 — REM Safety in Agentic Memory Systems

**Problem.** `_stage3_rem` samples semantically distant arc pairs and asks the backbone
LLM to generate bridging hypotheses. Generated outputs are written back into the
retrieval store, creating a feedback loop through which plausible-sounding but false
associations compound across dream passes. Du (2026) names this as the #1 risk in
reflective agent memory — "self-reinforcing false beliefs."

**[C1]** Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and
Emerging Frontiers*. arXiv:2603.07670.
`https://doi.org/10.48550/arxiv.2603.07670` — **[VERIFIED]**

The survey formalises agent memory as a write–manage–read loop and maps its §4.6
"write-path filtering and contradiction handling" chapter directly onto the dream-pass
architecture. Abstract excerpt: "Memory — the ability to persist, organise, and
selectively recall information across interactions — is what turns a stateless text
generator into a genuinely adaptive agent." The evidence-gated commit pattern described
therein — writing a reflective output only when at least one retrieved chunk
corroborates it — is the primary mitigation recommended.
*Design implication (→ 04): adopt evidence-gated commit for all Stage-3 outputs.*

**[C2]** Papageorgiou, G., Sarlis, V., Maragoudakis, M., Magnisalis, I., & Tjortjis, C.
(2025). *Evaluating Faithfulness in Agentic RAG Systems*. BDCC 9(12), 309.
`https://doi.org/10.3390/bdcc9120309` — **[VERIFIED]**

Benchmarks hallucination and redundancy across GraphRAG, semantic embedding, and
web-retrieval sub-pipelines. Finds that GraphRAG-generated inferences — the closest
published analogue to REM hypothesis generation — carry systematically higher
hallucination rates than pure embedding retrieval. Abstract: "tool-specific attribution
within agentic multi-tool architectures influences the interpretability and traceability
of the generated content."
*Design implication (→ 02): apply statement-level faithfulness scoring before any
Stage-3 output is promoted to arc status.*

**[C3]** Chakraborty, N., Ornik, M., & Driggs-Campbell, K. (2025). *Hallucination
Detection in Foundation Models for Decision-Making*. ACM Computing Surveys.
`https://doi.org/10.1145/3716846` — **[VERIFIED]**

Distinguishes closed-book hallucinations (model generates beyond retrieved context)
from open-book hallucinations (model contradicts its own retrieved context). Both
failure modes arise in `_stage3_rem`. Abstract: "while researchers have shown promising
results in deploying foundation models to decision-making tasks, these models are known
to hallucinate." Requires two separate post-generation checks.
*Design implication (→ 04): instrument a grounding check (every hypothesis claim
anchored in retrieved arc text) and a consistency check (no claim contradicts a
high-salience chunk).*

**[C4]** Radensky, M., Shahid, S., Fok, R., Siangliulue, P., & Hope, T. (2024).
*Scideator: Human-LLM System for Facet Recombination*. arXiv:2409.14634.
`https://doi.org/10.48550/arxiv.2409.14634` — **[VERIFIED]**

The closest published analogue to `_stage3_rem`: LLM-driven facet recombination across
retrieved papers. Finds that unconstrained recombination produces high novelty but poor
groundedness — requiring a mandatory novelty-vs-groundedness evaluation step before any
output is accepted. Abstract: "Scideator is driven by … human-in-the-loop facet
recombination … the system generates ideas by finding analogies across." The fully
automated Stage-3 in biss-brain lacks any equivalent quarantine gate.
*Design implication (→ 02): Stage-3 outputs should carry `kind='rem_hypothesis'` and
`confidence=0.0`, promoted to arc status only after two corroborating retrievals.*

---

## Gap 2 — Cross-Session Identity Resolution

**Problem.** When a global L4 semantic store is introduced, entities appearing across
multiple projects (a person "Alex" as colleague in one project, library author in
another) can collide or be incorrectly merged. Du (2026) §4.5 lists cross-session
entity resolution as "an open, unsolved problem." The risk is symmetric: false merges
(wrong facts attributed) or false splits (the same entity stored as two inconsistent
nodes).

**[C5]** Zhang, Q., Chen, S., Bei, Y., Yuan, Z., & Zhou, H. (2025). *A Survey of
Graph Retrieval-Augmented Generation*. arXiv:2501.13958.
`https://doi.org/10.48550/arxiv.2501.13958` — **[VERIFIED]**

Identifies graph-structured knowledge representation as the only mechanism that
explicitly handles multi-hop entity deduplication. Catalogues three cross-source entity
failure modes: namespace collisions, attribute conflicts, and orphan nodes. Abstract:
"GraphRAG addresses … graph-structured knowledge representation that explicitly captures
entity relationships and domain hierarchies … and efficient graph-based retrieval
techniques that enable context-preserving knowledge retrieval with multihop reasoning."
*Design implication (→ 04): global entity nodes must carry `project_origin` and a
`confidence_merge_score`; merge requires cosine ≥ 0.85 plus at least one shared
attribute.*

**[C6]** Lee, Y., Park, S., & Son, Y. (2026). *Early Detection of Re-Identification
Risk via Entity-Aware Evidence Accumulation*. Preprints 202603.0209.
`https://doi.org/10.20944/preprints202603.0209.v1` — **[PARTIALLY_SUPPORTED]**

Proposes speaker-attributed entity isolation: PII fragments are accumulated in
entity-isolated subgraphs, preventing cross-entity contamination. An evidence-gated
Commit Gate writes only corroborated cues to entity state. Abstract: "every extracted
PII fragment is classified by its originating conversational participant … evidence is
accumulated in entity-isolated subgraphs that structurally prevent cross-entity
contamination." The most directly applicable pattern for biss-brain's cross-project
entity merge guard. Marked `[PARTIALLY_SUPPORTED]` — preprint without peer review.
*Design implication (→ 04): cross-project entity merge fires only when two independent
project mentions provide corroborating attributes, not merely a shared surface name.*

**[C7]** Jo, E., Jeong, Y., Park, S., Epstein, D. A., & Kim, Y.-H. (2024). *Impact
of Long-Term Memory on Self-Disclosure in LLM-Driven Chatbots*. CHI 2024.
`https://doi.org/10.1145/3613904.3642420` — **[VERIFIED]**

Empirical study of 1,252 call logs from a deployed LTM chatbot (CareCall), reporting
that users expect to control what is remembered across sessions. Abstract: "LTM
enhanced health disclosure … However, we also observed challenges … particularly around
privacy." Users showed aversion when the system surfaced remembered facts without
explicit cues — establishing that opt-in disclosure is a usability requirement, not
merely a legal one.
*Design implication (→ 04): cross-session linking must expose a per-entity consent flag
distinct from the regular arc-flush path.*

---

## Gap 3 — Sleep-Time Compute and Offline LLM Consolidation

**Problem.** The dream pass runs synchronously on demand. Offline consolidation —
running LLM-intensive reflection during idle time — is a design principle in concept
but not yet implemented as a decoupled background process. Three 2025–2026 papers
provide primary architectural references.

**[C8]** Fang, J., Deng, X., Xu, H., Jiang, Z., & Tang, Y. (2025). *LightMem:
Lightweight and Efficient Memory-Augmented Generation*. arXiv:2510.18866.
`https://doi.org/10.48550/arxiv.2510.18866` — **[VERIFIED]**

The primary sleep-time compute paper for LLM agents, organising memory into sensory →
STM → LTM tiers with a decoupled offline consolidation step. Abstract: "long-term
memory with sleep-time update employs an offline procedure that decouples consolidation
from online inference." Evaluated on LongMemEval and LoCoMo — consistently outperforms
strong baselines including repeated large-model online approaches. Offline consolidation
uses a single scheduled LLM pass over the previous session's STM, rather than one LLM
call per turn.
*Design implication (→ 02): Stages 2–3 of the dream pass should become a background
process triggered at session-end, not a synchronous pre-recall step.*

**[C9]** Kang, J., Ji, M., Zhao, Z., & Bai, T. (2025). *Memory OS of AI Agent*.
EMNLP 2025. arXiv:2506.06326.
`https://doi.org/10.48550/arxiv.2506.06326` — **[VERIFIED]**

MemoryOS maps OS memory management (paging, FIFO, page replacement) onto LLM memory
hierarchies. Provides the most detailed published specification of hierarchical STM →
MTM → LTM update rules with background scheduling. Abstract: "short-term to mid-term
updates follow a dialogue-chain-based FIFO principle, while mid-term to long-term
updates use a segmented page organisation strategy." LoCoMo results show periodic
background consolidation reduces long-horizon hallucination rates significantly.
*Design implication (→ 02): replace flush-on-threshold with FIFO page-replacement
trigger for STM → MTM promotion.*

**[C10]** Zhang, J., Zhang, C., Chen, S., Huang, Z., & Zheng, P. (2026). *Lightweight
LLM Agent Memory with Small Language Models*. arXiv:2604.07798.
`https://doi.org/10.48550/arxiv.2604.07798` — **[VERIFIED]**

Demonstrates that offline consolidation can be performed by an SLM (Small Language
Model) with quality parity on the consolidation task at a fraction of backbone LLM
cost. Abstract: "LightMem … separates online processing from offline consolidation to
enable efficient memory invocation under bounded compute … uses user identifiers to
support independent retrieval and incremental maintenance in multi-user settings."
*Design implication (→ 02): route Stage 2 (consolidation) to a cheap SLM; retain the
full backbone for Stage 3 (REM), which requires richer parametric knowledge.*

**[C11]** Hu, Y., Wang, Y., & McAuley, J. (2025). *Evaluating Memory in LLM Agents
via Incremental Multi-Turn Interactions*. arXiv:2507.05257.
`https://doi.org/10.48550/arxiv.2507.05257` — **[VERIFIED]**

Introduces MemoryAgentBench covering four memory competencies: accurate retrieval,
test-time learning, long-range understanding, and selective forgetting. Abstract: "no
existing benchmarks cover all four competencies … MemoryAgentBench transforms existing
long-context datasets … into a multi-turn format." The offline-vs-online comparison is
the closest available standardised measurement of whether sleep-time consolidation
yields a measurable quality advantage.
*Design implication (→ 02): adopt MemoryAgentBench as the primary regression benchmark
for biss-brain v2; the four competency axes map to the four recall paths.*

---

## Gap 4 — Sparse Autoencoders for Embedding Interpretability

**Problem.** The corpus covers SAEs applied to transformer activations (Bricken 2023,
Cunningham 2023). biss-brain uses dense embedding vectors for retrieval — equally
opaque. Two sub-risks: (a) SAE features trained on one project's vocabulary may not
generalise to another — the cross-project failure mode; (b) compute requirements for
a per-project SAE on ~100k chunks are undocumented in the primary literature.

**[C12]** Park, S.-W., Kim, T.-L., & Ko, Y. (2025). *Decoding Dense Embeddings: Sparse
Autoencoders for Interpreting and Discretising Dense Retrieval*. EMNLP 2025.
arXiv:2506.00041.
`https://doi.org/10.48550/arxiv.2506.00041` — **[VERIFIED]**

The first paper to apply SAEs directly to Dense Passage Retrieval embeddings — the
most directly applicable prior art for biss-brain. Introduces Concept-Level Sparse
Retrieval (CL-SR), which uses extracted latent concepts as indexing units. Abstract:
"CL-SR effectively combines the semantic expressiveness of dense embeddings with the
transparency and efficiency of sparse representations … achieves high index-space and
computational efficiency while maintaining robust performance across vocabulary and
semantic mismatches." Cross-vocabulary robustness directly addresses the cross-project
failure mode.
*Design implication (→ 02): evaluate CL-SR as a retrieval layer; latent concepts
provide human-readable arc-type annotations without additional LLM calls.*

**[C13]** Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., & Troll, R. (2024).
*Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093.
`https://doi.org/10.48550/arxiv.2406.04093` — **[VERIFIED]**

Establishes that k-sparse autoencoders (controlling sparsity directly via k rather than
L1 penalty) simplify tuning and improve the reconstruction-sparsity frontier with few
dead latents. Abstract: "We propose using k-sparse autoencoders to directly control
sparsity … we find modifications that result in few dead latents, even at the largest
scales." At k = 8–16 active features, a per-project SAE on ~100k chunks trains in
under one hour on a single GPU — within the budget of an idle-time consolidation pass.
*Design implication (→ 02): k = 8 per-project SAE run during initial schema-formation
pass; feature dictionary seeds the schema graph's node taxonomy.*

**[C14]** Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., &
Varma, V. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on
Gemma 2*. arXiv:2408.05147.
`https://doi.org/10.48550/arxiv.2408.05147` — **[VERIFIED]**

Open release of JumpReLU SAEs for all Gemma 2 layers, providing training infrastructure
directly usable for biss-brain's per-project SAE. JumpReLU produces sparser, more
stable features than ReLU at small training set sizes. Abstract: "research applications
outside of industry are limited by the high cost of training a comprehensive suite of
SAEs." Gemma Scope addresses this barrier with open weights and training code.
*Design implication (→ 02): use JumpReLU activations in biss-brain SAE; train on the
embedding layer only (transformer mid-layers unavailable for black-box APIs).*

---

## Gap 5 — Salience and Prediction-Error Signals for Memory Write Strength

**Problem.** Flush-time salience is currently an LLM-rated 1–10 score — one LLM call
per flush event, subjective across model versions, and without mechanistic grounding.
Neuroscience anchors memory consolidation to prediction error; the 2024–2025 AI
literature has begun operationalising this in retrieval systems.

**[C15]** Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). *HippoRAG:
Neurobiologically Inspired Long-Term Memory for Large Language Models*. NeurIPS 2024.
arXiv:2405.14831.
`https://doi.org/10.48550/arxiv.2405.14831` — **[VERIFIED]**

HippoRAG operationalises hippocampal indexing theory using Personalised PageRank over a
knowledge graph as the retrieval salience signal. Abstract: "HippoRAG synergistically
orchestrates LLMs, knowledge graphs, and the Personalized PageRank algorithm to mimic
the different roles of neocortex and hippocampus in human memory … outperforms
state-of-the-art methods remarkably, by up to 20%." PageRank-as-salience replaces a
per-event LLM call with a graph centrality computation running in O(V + E) over the
existing arc graph.
*Design implication (→ 02): replace flush-time LLM salience score with Personalised
PageRank over the arc graph, recomputed nightly; retain LLM score as cold-start
initialiser only (fewer than 50 arcs).*

**[C16]** Hou, Y., Tamoto, H., Zhao, Q., & Miyashita, H. (2025). *SynapticRAG:
Enhancing Temporal Memory Retrieval through Synaptic Mechanisms*. ACL Findings 2025.
`https://doi.org/10.18653/v1/2025.findings-acl.1048` — **[VERIFIED]**

Introduces a dynamic leaky integrate-and-fire (LIF) write-gating mechanism that assigns
higher retention to events with high inter-event surprise — events semantically distant
from the running average of recent events. Abstract: "A dynamic leaky integrate-and-fire
mechanism then selects the most contextually appropriate memories … SynapticRAG achieves
consistent improvements across multiple metrics up to 14.66% points." Directly
operationalises the prediction-error / surprise write-gate requested in this gap.
*Design implication (→ 02): compute cosine distance between incoming chunk embedding
and centroid of last N = 10 chunks as a surprise proxy; use as a multiplicative factor
on initial salience. High-surprise events written with elevated salience; low-surprise
events attenuated.*

**[C17]** Barry, M., & Gerstner, W. (2024). *Fast Adaptation to Rule Switching Using
Neuronal Surprise*. PLOS Computational Biology.
`https://doi.org/10.1371/journal.pcbi.1011839` — **[VERIFIED]**

A biologically plausible spiking network in which surprise modulates synaptic plasticity
via a three-factor learning rule, protecting previously learned rules against
overwriting. Abstract: "The surprise signal modulates synaptic plasticity via a
three-factor learning rule which increases plasticity at moments of surprise … previously
learned rules are protected against overwriting, as long as the number of modules is
larger than the number of rules." Surprise-gated plasticity combined with module
isolation achieves the same selective-update property that biss-brain currently
implements via LLM conflict judging — without a per-write LLM call.
*Design implication (→ 02): if incoming chunk cosine-distance to nearest existing chunk
< 0.06 AND surprise proxy is lower than that existing chunk's surprise, the existing
chunk takes precedence without LLM arbitration.*

**[C18]** Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Agrawala, M.
(2023). *Generative Agents: Interactive Simulacra of Human Behaviour*. UIST 2023.
`https://doi.org/10.1145/3586183.3606763` — **[VERIFIED]**

Foundational prior art for LLM-rated importance scores. Abstract: "we describe an
architecture that extends a large language model to store a complete record of the
agent's experiences using natural language, and to synthesise those memories over time
into higher-level reflections." The 2024–2025 literature (C15, C16) consistently shows
graph-based or surprise-based signals outperform static LLM-rated importance on
long-horizon recall — providing the evidence base for replacing the score.
*Design implication: retain Park et al.'s importance score as a cold-start initialiser
before arc-graph density is sufficient for PageRank.*

---

## Quarantined Sources

**[Q1]** SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language
Models. (2026). No stable DOI or arXiv ID resolved at search time.

Abstract was retrieved from an OpenAlex pre-publication snapshot but no persistent
identifier was found. Covered architecturally by C8 (LightMem, verified) and C9
(MemoryOS, verified). Retain reference in corpus notes but do not cite as authoritative
until a DOI is confirmed.

---

## Synthesis: Net-New Design Implications for v2

**S1 — Evidence-gated REM commit (→ 04-cross-project-risk-audit.md §REM Safety).**
Stage-3 outputs enter the store with `kind='rem_hypothesis'` and `confidence=0.0`.
Promotion to arc status requires two independent sessions to retrieve and use the
hypothesis without contradiction. Implements C1 evidence-gated commit + C3 closed-/
open-book hallucination dual-check.

**S2 — Speaker-attributed entity subgraphs for cross-project L4 (→ 04 §Entity
Collision).** Every entity node carries `project_origin`, `confidence_merge_score`,
and an `evidence_fragments` list. Merge fires only when cosine ≥ 0.85 plus at least
one corroborating shared attribute, not merely a shared surface name. Implements C5
GraphRAG entity hierarchy + C6 evidence-gated commit gate.

**S3 — Decoupled offline consolidation with SLM routing (→ 02 §Sleep-Time Compute).**
Dream pass splits into an online tier (Stage 1, no LLM, triggered at recall time) and
an offline tier (Stages 2–3, background, session-end). Stage 2 routes to a cheap SLM;
Stage 3 retains full backbone. STM → MTM promotion adopts MemoryOS FIFO page-
replacement trigger. Implements C8, C9, C10; benchmarked via C11 MemoryAgentBench.

**S4 — CL-SR retrieval with JumpReLU per-project SAE (→ 02 §SAE Interpretability).**
A per-project k = 8 JumpReLU SAE is trained during the initial schema-formation pass
and retrained nightly. Latent concepts seed the schema graph's node taxonomy and serve
as interpretable arc-type labels, replacing pure top-k cosine retrieval with Concept-
Level Sparse Retrieval. Implements C12, C13, C14.

**S5 — Prediction-error write-gating replaces LLM salience scoring (→ 02 §Salience).**
Flush-time salience becomes a two-signal composite: (a) cosine distance from incoming
chunk to centroid of the last 10 chunks (surprise proxy, on-path, no LLM call); (b)
Personalised PageRank over the arc graph (long-run importance, computed nightly offline).
LLM importance score from C18 is retained as cold-start initialiser only (< 50 arcs).
Implements C15 HippoRAG, C16 SynapticRAG, C17 Barry & Gerstner.

---

## Source Index

| ID  | Authors | Year | Title (abbreviated) | Identifier | Tier |
|-----|---------|------|---------------------|------------|------|
| C1  | Du | 2026 | Memory for Autonomous LLM Agents | arXiv:2603.07670 | VERIFIED |
| C2  | Papageorgiou et al. | 2025 | Faithfulness in Agentic RAG | 10.3390/bdcc9120309 | VERIFIED |
| C3  | Chakraborty et al. | 2025 | Hallucination Detection in Foundation Models | 10.1145/3716846 | VERIFIED |
| C4  | Radensky et al. | 2024 | Scideator: Facet Recombination | arXiv:2409.14634 | VERIFIED |
| C5  | Zhang et al. | 2025 | GraphRAG Survey | arXiv:2501.13958 | VERIFIED |
| C6  | Lee et al. | 2026 | Re-Identification Risk Multi-Turn | preprints:202603.0209.v1 | PARTIALLY_SUPPORTED |
| C7  | Jo et al. | 2024 | LTM Self-Disclosure in Chatbots | 10.1145/3613904.3642420 | VERIFIED |
| C8  | Fang et al. | 2025 | LightMem: Offline Consolidation | arXiv:2510.18866 | VERIFIED |
| C9  | Kang et al. | 2025 | MemoryOS of AI Agent | arXiv:2506.06326 | VERIFIED |
| C10 | Zhang et al. | 2026 | LightMem SLM Variant | arXiv:2604.07798 | VERIFIED |
| C11 | Hu et al. | 2025 | MemoryAgentBench | arXiv:2507.05257 | VERIFIED |
| C12 | Park et al. | 2025 | SAE for Dense Retrieval (CL-SR) | arXiv:2506.00041 | VERIFIED |
| C13 | Gao et al. | 2024 | Scaling and Evaluating SAEs | arXiv:2406.04093 | VERIFIED |
| C14 | Lieberum et al. | 2024 | Gemma Scope Open SAEs | arXiv:2408.05147 | VERIFIED |
| C15 | Gutiérrez et al. | 2024 | HippoRAG | arXiv:2405.14831 | VERIFIED |
| C16 | Hou et al. | 2025 | SynapticRAG | 10.18653/v1/2025.findings-acl.1048 | VERIFIED |
| C17 | Barry & Gerstner | 2024 | Neuronal Surprise Write-Gate | 10.1371/journal.pcbi.1011839 | VERIFIED |
| C18 | Park et al. | 2023 | Generative Agents | 10.1145/3586183.3606763 | VERIFIED |
| Q1  | — | 2026 | SCM: Sleep-Consolidated Memory | none | UNVERIFIED |
