# Bio-Inspired AI Memory Architectures: A Skeptical Survey

> Companion to `01-neuroscience-of-memory.md`. The neuroscience tells us *what* the
> brain does; this document records *which AI primitives* have been proposed to
> implement those mechanisms, and which actually work in production.

---

## 1. Hierarchical Temporal Memory (HTM) — Numenta / Hawkins

**What it is / biological mapping.** HTM models the neocortex's columnar
microstructure. The Spatial Pooler converts input into a sparse distributed
representation (SDR) — a high-dimensional binary vector with roughly 2% active bits —
analogous to how cortical columns encode sensory information. The Temporal Memory layer
learns sequences over those SDRs, mimicking predictive firing in layer 5 pyramidal
neurons. Hawkins' 2021 *Thousand Brains* extends this: every cortical column
independently builds a complete world model using reference frames (grid-cell-like
structures), and perception emerges from a voting mechanism across columns.

**Concrete algorithm.** Spatial Pooler: overlap-based competition + Hebbian boosting to
produce fixed-sparsity output. Temporal Memory: each minicolumn has multiple cells;
context from prior SDRs predicts which cell becomes active. SDRs support set-theoretic
operations (union, overlap score) that are noise-tolerant by design.

**Production reality.** Numenta published an open-source Python library (`htm.core`,
formerly `nupic`) and in 2024 launched the Thousand Brains Project with "Monty," a
sensorimotor learning framework. Applications are narrow: anomaly detection in
time-series (e.g., IoT sensor streams) is the one domain where HTM demonstrably
outperforms simple baselines. There is **no published evidence HTM is used in any
production LLM or RAG pipeline**. The "Thousand Brains" reference frame theory is
mathematically underdeveloped — Numenta has not produced a scalable training algorithm
for the grid-cell cortical column architecture.

**Verdict.** SDRs are a genuinely useful encoding primitive (noise tolerance, set
overlap as similarity). The full HTM stack is a research curiosity, not an engineering
tool. **Steal SDRs; ignore the rest.**

---

## 2. Modern (Continuous) Hopfield Networks

**What it is / biological mapping.** Ramsauer et al. (2021, ICLR — arXiv 2008.02217)
showed that replacing the classical binary Hopfield energy with a continuous
log-sum-exp energy produces a network whose update rule is **exactly the softmax
attention operation used in Transformers**. Storage capacity scales exponentially with
embedding dimension (2^(d/2) patterns), versus O(d) for classical Hopfield. One
retrieval step converges to a stored pattern. Maps to associative memory in
hippocampus CA3 / cortical layer activation patterns.

**Concrete algorithm.** Given a query vector ξ and stored patterns X (the key matrix),
one update step is:

```
ξ_new = X · softmax(β · X^T · ξ)
```

This is dot-product attention. The JKU group published `hopfield-layers` (PyTorch),
which wraps this as drop-in attention alternatives, pooling operators, or lookup
modules.

**Practical relevance for RAG.** Standard top-k cosine retrieval is already isomorphic
to a one-step Hopfield retrieval with temperature β → ∞. The interesting extensions:

- finite-temperature Hopfield retrieval returns a **weighted blend** of patterns
  rather than a single result — useful for soft associative lookup;
- the Adaptive Hopfield variant (arXiv Nov 2024) replaces cosine with a learned
  similarity that better approximates retrieval likelihood;
- Hopfield-Fenchel-Young Networks (Nov 2024) generalise the energy function further.

**Production reality.** `hopfield-layers` exists and works. It has been used in
bioinformatics (immune repertoire classification) and tabular data (`hopular`). Direct
use as a RAG retrieval layer over embedding stores has not been demonstrated at
production scale. The conceptual payoff is clear: instead of returning the single
nearest neighbour, a Hopfield layer over your embedding matrix returns a
probability-weighted blend — a soft associative lookup.

**Verdict.** Implementable today as a five-line addition over standard top-k cosine.
**Use as the final blending step of the retrieval pipeline.**

---

## 3. Memory-Augmented Neural Networks (NTM / DNC)

**What it is / biological mapping.** Graves et al. NTM (2014, arXiv 1410.5401) adds an
external memory matrix M of shape (N×d) to an RNN controller. Read/write heads attend
to memory locations using content-based (cosine) and location-based addressing. The
DNC (Graves et al., 2016, *Nature*) adds temporal link matrices tracking write order
and a usage vector for memory allocation. Maps to hippocampal working memory and the
prefrontal-hippocampal loop.

**Lessons learned.** NTMs and DNCs were never production-grade. Training is notoriously
unstable (NaN gradients documented in open-source implementations). The soft attention
over all N memory slots scales quadratically. The DNC could solve algorithmic tasks
(sorting, graph traversal) that pure RNNs failed, but these tasks are trivially solved
by conventional programs.

The transformative insight they produced was not the architecture itself but the
conceptual separation of a controller from an external addressable memory — which
survives directly into modern LLM agent architectures (tool calls to vector stores,
SQL, etc.).

**Verdict.** NTM / DNC are intellectual ancestors. **Do not implement them.** Their
lessons live on in the design of read/write APIs for agent memory stores.

---

## 4. Replay-Based Continual Learning

**What it is / biological mapping.** Biological sleep involves hippocampal replay —
compressed replaying of waking experience to consolidate into neocortical weights.
Computational analogues:

- **Experience Replay** (Lin 1992; DQN, Mnih 2015): buffer of past transitions,
  sampled during training.
- **Generative Replay** (Shin et al. 2017, NeurIPS): a generator (GAN/VAE) hallucinates
  past data; the main model trains on real + synthetic.
- **Dark Experience Replay / DER++** (Buzzega et al. 2020, NeurIPS): stores model
  logits (soft targets) alongside inputs; training minimises KL to past logits —
  essentially distilling prior model knowledge into the new model via the buffer.
- **Sleep-inspired GAN replay** (Tadros et al. 2020): GAN replays are triggered during
  a simulated sleep phase.

**Replay vs. regularisation.** The empirical consensus (van de Ven et al. 2024 TPAMI
survey) is that replay methods consistently beat EWC-style parameter regularisation
(Kirkpatrick 2017, *PNAS*) under Class-IL (new classes added) scenarios. EWC works only
when task boundaries are sharp and the Fisher matrix is accurate — conditions that
rarely hold in practice. **DER++ is the best simple baseline**: cheap, no generative
model needed, strong empirical results.

**For an LLM agent system, the practical analog is:** store (input, prior model output)
pairs and periodically retrain or fine-tune against them. This is exactly what
"sleep-time compute" papers are rediscovering.

**Verdict.** A SQLite table of (timestamp, episode_embedding, prior_response_logits)
functions as an experience buffer. Periodic fine-tuning or in-context distillation
against this buffer is your replay mechanism. **Keep on the radar; implement when
fine-tuning loop is wanted.**

---

## 5. Predictive Coding / Free Energy Principle

**What it is / biological mapping.** Rao & Ballard (1999, *Nature Neuroscience*)
proposed that each cortical layer maintains a generative model of the layer below;
prediction errors propagate upward. Friston extended this to the Free Energy
Principle: the brain minimises variational free energy, which bounds surprise. Every
perception is inference, every action is policy optimisation.

**Concrete algorithm.** Predictive coding networks (PCNs) have a two-phase forward
pass: (1) feedforward sweep (fast, amortised) produces an initial prediction, then
(2) iterative inference corrects local prediction errors layer by layer until
convergence. Millidge, Tschantz & Buckley formalised that PCN inference approximates
backpropagation (*Neural Computation* 2022). Their Hybrid Predictive Coding paper
(Tschantz et al., *PLOS CompBio* 2023) combines fast amortised + slow iterative
inference; code at <https://github.com/alec-tschantz/pybrid>.

**Practical relevance for memory.** If your system maintains a generative model of
"what the codebase looks like," retrieval becomes inference (minimise prediction error
against stored patterns) rather than lookup. Beautiful in theory.

In practice, training a PCN competitive with a standard transformer requires
substantial infrastructure and yields no demonstrated advantage on coding tasks. The
FEP literature is philosophically rich but empirically thin on practical ML benchmarks.

**Verdict.** Intellectually formative for understanding why prediction errors are the
right currency for memory updates. **Not implementable at production scale without
significant research investment.**

---

## 6. Vector-Symbolic Architectures / Hyperdimensional Computing

**What it is / biological mapping.** Kanerva (2009, *Cognitive Computation*) formalised
computing with high-dimensional random vectors (d ≈ 10,000). Three operations:

- **binding** (element-wise multiplication / XOR — produces a dissimilar vector
  encoding the association of two concepts)
- **bundling** (element-wise addition / majority — produces a superposition
  representing a set)
- **permutation** (cyclic shift — encodes sequence/role)

Maps to neural population codes and distributed representations across cortical areas.

**Concrete algorithm.** To store "key→value": compute `bound = bind(key, value)`; add
to a bundle (the "memory"). To query: compute `query = bind(key, bundle)`; cosine
similarity to value vocabulary recovers the stored value. Kleyko, Rachkovskij, Osipov
& Rahimi (*ACM Computing Surveys*, 2023) is the canonical modern survey. FPGA
implementations achieve real-time inference with minimal power.

**Practical relevance.** VSA operations are O(d) and embarrassingly parallel. As a
memory primitive over embeddings: you can use binding to encode relational triples
(subject, predicate, object) into a single vector, then bundle thousands of them into
one "memory vector" that supports approximate associative lookup. This is a legitimate
alternative to a graph database for fixed-schema relational facts.

The downside: capacity is limited (~d/log(d) distinct bindings before interference),
and modern embedding spaces are not designed for VSA operations — alignment with
BERT-style embeddings requires careful normalisation.

**Verdict.** Interesting for a typed-edge encoding layer if/when graph storage becomes
a bottleneck. **Defer.**

---

## 7. Episodic-to-Semantic Distillation in LLM Agents

**Generative Agents** (Park et al., 2023, UIST) — every observation gets a timestamp,
an embedding, and a salience score (LLM-rated 1–10). Retrieval scores =
recency (exponential decay) + cosine similarity + importance. Periodic *reflection*:
the agent clusters recent high-importance memories and generates higher-order semantic
summaries. **This is the canonical episodic→semantic pipeline.**

**A-MEM** (Xu et al., NeurIPS 2025, arXiv 2502.12110) — Zettelkasten-style. Each memory
note carries keywords, tags, contextual description, and backlinks to related notes.
New memories trigger dynamic re-linking. More structured than Park et al., better at
handling evolving knowledge graphs. **Caveat:** untyped similarity links cause
contextually-unrelated memories to collapse together (the canonical failure mode for
embedding-based linking).

**Sleep-time compute** — an emerging cluster of 2025 papers (LightMem, SCM:
Sleep-Consolidated Memory, SleepGate, "Language Models Need Sleep") operationalises
offline consolidation. LightMem reports up to 10.9% accuracy gains with 117× fewer
tokens via sleep-time update. SCM uses multi-dimensional importance tagging +
intentional forgetting.

**Chain-of-thought as inner replay** — not a paper but an observation confirmed by
several empirical analyses: CoT scratchpads function as working memory replay,
improving multi-step reasoning by the same mechanism that inner speech supports human
working memory.

**Verdict.** The Park et al. reflection architecture is implementable in a weekend on
top of SQLite + a vector store + an LLM call. **Importance scoring + embedding +
recency decay retrieval is the minimum viable episodic memory.** This is largely
already shipped in biss-brain's dream pass.

---

## 8. Modern Hopfield as Retrieval Architecture (Hopfield-as-RAG)

**HippoRAG** (Bernal Jiménez Gutiérrez et al., 2024, NeurIPS — arXiv 2405.14831)
takes a different
biologically-inspired path: it builds a hippocampal-index knowledge graph from
documents and uses Personalized PageRank (PPR) for retrieval. Results: +11–20% on
multi-hop QA (2WikiMultiHopQA) vs. standard RAG; single-step retrieval matches
iterative methods at 10–30× lower cost.

The Hopfield framing clarifies *why* this works: PPR over a graph is an iterative
attractor-like relaxation that allows partial cues to complete full memory traces —
the CA3 pattern completion mechanism in software.

**Adaptive Hopfield Network** (arXiv Nov 2024) replaces cosine similarity in the
retrieval kernel with a learned similarity function approximating retrieval likelihood.
Still in preprint; no production results.

**Verdict.** HippoRAG is real and production-adjacent (open-source, reproducible).
Pure Hopfield layers as a drop-in retrieval replacement for cosine ANN are
theoretically sound but not yet demonstrated to outperform cosine ANN in practice at
scale. **Use HippoRAG-style PPR over the schema graph for multi-hop questions; use
Hopfield soft blend as a final post-retrieval step.**

---

## 9. Sparse Coding and Dictionary Learning

**Olshausen & Field (1996, *Nature*)** showed that imposing sparsity constraints on
representations of natural images recovers Gabor-like V1 simple-cell receptive fields.
Dictionary learning generalises this: learn an overcomplete basis D such that x ≈ D·a
where a is sparse.

**Anthropic SAE work.** Bricken et al. (2023, "Towards Monosemanticity") and
Cunningham et al. (2023, arXiv 2309.08600) train sparse autoencoders on residual
stream activations. The SAE learns an overcomplete dictionary of ~tens of thousands
of "features" that are monosemantic (single concept per feature). Scaling
Monosemanticity (2024) extends this to Claude 3 Sonnet with millions of features.

**Can SAE features serve as memory neurons?** Mechanistically yes: SAE features are
sparse, interpretable, and compositional — closer to grandmother-cell-style
representations than dense embeddings. An SAE trained on your codebase's LLM
activations would produce a dictionary of code concepts (e.g., "async error handling,"
"SQL injection pattern") as discrete features. Retrieval could then match sparse
feature vectors rather than dense embeddings, gaining both interpretability and
potentially better precision for rare concepts.

The engineering gap: SAE training requires large volumes of forward-pass activations
(~billions of tokens), which is feasible for a large codebase but non-trivial to set
up. Open-source tooling exists (`dictionary_learning` by Samuel Marks et al.,
`eleutherai/sae`).

**Verdict.** Promising for v1.0 (research-grade). **For v0.x, use top-k binarisation
of dense embeddings as a poor-man's sparse signature.**

---

## 10. Spiking Neural Networks / Neuromorphic Compute

SNNs communicate via discrete spikes with timing-dependent plasticity (STDP). Intel
Loihi 2 / IBM TrueNorth / Hala Point are neuromorphic chips that run SNNs at
radically lower power than GPUs for sparse computation.

**Application-layer reality.** At the software layer, SNNs are uniformly worse than
standard ANNs on the tasks relevant to a coding assistant (language modelling, code
retrieval, semantic search). The energy advantage is real on neuromorphic hardware
but vanishes when simulated on GPUs. STDP as a learning rule cannot train
classification layers without supervision; hybrid approaches are required. Hala Point
(2024) demonstrates 1/150th GPU energy for sparse video processing — a hardware
advantage only.

**Verdict.** SNNs are a hardware story, not a software-layer primitive. Unless
deploying on Loihi 2, **ignore SNNs**. The STDP *concept* (timing-aware Hebbian
update) is still useful as a software heuristic — see the co-retrieval edge bumps in
the architecture spec.

---

## 11. Schema-Based / Graph-Augmented Memory

**Microsoft GraphRAG** (Edge et al., 2024). LLM extracts entities and relationships
from a corpus → knowledge graph → Leiden community detection → hierarchical community
summaries used for retrieval. Excels at "global sensemaking" queries (e.g., "what are
the main themes?") but is expensive (many LLM calls per ingestion), slow to update,
and brittle to schema drift.

**Graphiti** (Zep). Temporally-aware knowledge graph with continuous ingestion. Uses
LLM-based entity resolution + name matching. Known limitation: entity resolution
degrades as entity vocabulary grows — the same real-world object may appear as
multiple graph nodes.

**Schema evolution.** Neither system solves this cleanly. GraphRAG's community
structure is recomputed from scratch on updates. Graphiti's temporal edges help track
fact changes but not ontology changes. Entity resolution via LLM is inherently
probabilistic and the error rate is not bounded.

**Verdict.** Graph memory is irreplaceable for relational facts that span documents
(cross-file dependencies, author/file/change relationships in a codebase). **Use it
alongside, not instead of, a vector store.** For a coding assistant: a lightweight
graph (NetworkX + SQLite) with LLM-extracted edges for "file imports module,"
"function calls function," "PR modified file" is low-cost and high-value. **Avoid
full GraphRAG** for dynamic codebases — ingestion cost too high.

---

## 12. Attractor Dynamics Over Embedding Stores (CA3-Style Pattern Completion)

CA3 is a recurrent network with strong recurrent collaterals. When given a partial cue
("I remember something about error handling in the auth module..."), attractor
dynamics iteratively complete the pattern to a stored memory. The analog in RAG:
iterative retrieval — retrieve top-k, expand with linked context, re-embed and
retrieve again.

**Does anyone actually do this?** Yes:

- **IRCoT** (Trivedi et al., 2022): interleaves chain-of-thought reasoning steps with
  retrieval — multi-hop retrieval as iterative attractor relaxation.
- **HippoRAG + PPR**: PPR is mathematically an attractor process over the graph —
  random walk converges to a distribution peaked on relevant nodes.
- **GAAMA** (arXiv 2603.27910): Graph-Augmented Associative Memory for Agents —
  graph traversal + associative retrieval.
- **Predictive Associative Memory** (arXiv 2602.11322, Feb 2025): retrieval by
  temporal co-occurrence rather than representational similarity — memories linked by
  "what happened next" rather than "what looks similar."

**Implementation.** Iterative retrieval is straightforward: retrieve top-k → extract
keyphrases/entities → retrieve again using those. Two or three hops typically saturate
recall. More than that and you're chasing noise.

**Verdict.** **Use as the recall pipeline's expansion step.** The CA3 framing
justifies why this works: you're allowing retrieval to settle into an attractor
rather than accepting the nearest neighbour's answer immediately.

---

## Architectural Primitives That Compose Well

After surveying the landscape skeptically, the following subset composes into a
coherent and implementable architecture for a coding assistant second brain:

**Layer 1 — Episodic store.** SQLite + LanceDB per project for raw chunks with
salience and embedding. **Hippocampus.** Already built.

**Layer 2 — Semantic consolidation (sleep-time compute).** Scheduled offline reflection
(Park et al. style), DER++-style logits buffer for replay. **NREM-3 + replay.**

**Layer 3 — Schema graph.** Lightweight typed-edge graph (HippoRAG-style) for
relational queries. **mPFC scaffold.**

**Layer 4 — Hopfield soft lookup.** Five-line softmax-blend over the top-k.
**CA3 pattern completion.**

**Sparse signatures via top-k binarisation now; SAE features later (v1.0).**
**DG pattern separation.**

**What to avoid.** HTM full stack (no production path), NTM/DNC (superseded by LLM
tool use), predictive coding networks (research-stage, requires training
infrastructure), SNNs (hardware story only), VSA over dense embeddings (impedance
mismatch), full GraphRAG community summaries (ingestion cost too high).

Sparse autoencoders are promising for interpretability and sparse-feature retrieval
but require significant upfront training investment — defer unless interpretability is
a first-class requirement.

---

## Source Index

- HTM: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5712570/>
- Hopfield Networks is All You Need (Ramsauer 2020): <https://arxiv.org/abs/2008.02217>
- hopfield-layers: <https://github.com/ml-jku/hopfield-layers>
- NTM (Graves 2014): <https://arxiv.org/abs/1410.5401>
- DER++ (Buzzega 2020): <https://arxiv.org/abs/2004.07211>
- Continual Learning Survey: <https://arxiv.org/html/2403.05175v1>
- Hybrid Predictive Coding: <https://arxiv.org/abs/2204.02169>
- Towards Monosemanticity (Anthropic): <https://transformer-circuits.pub/2023/monosemantic-features>
- Scaling Monosemanticity: <https://transformer-circuits.pub/2024/scaling-monosemanticity/>
- Sparse Autoencoders Find Highly Interpretable Features (Cunningham 2023): <https://arxiv.org/abs/2309.08600>
- HippoRAG (Bernal Jiménez Gutiérrez 2024): <https://arxiv.org/abs/2405.14831>
- GraphRAG (Microsoft): <https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/>
- Graphiti (Zep): <https://github.com/getzep/graphiti>
- A-MEM (Xu 2025): <https://arxiv.org/abs/2502.12110>
- VSA Survey: <https://dl.acm.org/doi/10.1145/3558000>
- Predictive Associative Memory: <https://arxiv.org/html/2602.11322v1>
- SCM (Sleep-Consolidated Memory): <https://arxiv.org/html/2604.20943v1>
- LightMem: <https://arxiv.org/html/2510.18866v1>
- LLM Agent Memory Survey: <https://arxiv.org/html/2603.07670v1>
- GAAMA: <https://arxiv.org/html/2603.27910>
- Numenta SDR theory: <https://discourse.numenta.org/t/properties-of-sparse-distributed-representations-and-their-application-to-hierarchical-temporal-memory/577>
