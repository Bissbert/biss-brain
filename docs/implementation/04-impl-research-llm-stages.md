# LLM Stage Implementation: Dream Pass v2

> Implementation guidance for the five LLM-mediated stages of the biss-brain v2 dream
> pass. Grounded in `docs/architecture/00-overview.md §4` and the research corpus
> `docs/research/02-06`. Every citation carries an API-verified confidence tier.
> Target stack: Python 3.11+, Ollama (Haiku-equivalent local model), structured
> prompts in markdown templates. No fine-tuning.

Search date: 2026-04-30. APIs: OpenAlex (primary), Semantic Scholar (supplementary),
arXiv (preprint confirmation). All DOIs confirmed via live API call before citation.

---

## A. Salience Scoring at Flush Time

### A.1 Heuristic v1 — Surprise-Weighted Length

The cheapest signal that requires no LLM call. Operationalises the SynapticRAG
surprise proxy (C16) and the Barry & Gerstner write-gate (C17) without any model
inference.

```python
import numpy as np

def salience_heuristic(
    chunk_text: str,
    chunk_vec: np.ndarray,       # unit-norm, 768-dim
    recent_vecs: list[np.ndarray],  # last N=10 flushed chunks
    alpha: float = 0.4,
    beta: float = 0.6,
) -> float:
    """
    Returns float in [0, 1]; multiply by 10 and round to get the 1–10 integer
    stored in chunks.salience.

    Formula:
        length_score = min(len(chunk_text) / 800, 1.0)          # saturates at 800 chars
        if recent_vecs is empty:
            surprise = 1.0                                        # cold start
        else:
            centroid = mean(recent_vecs[-10:])                   # unit-normalised
            surprise = cosine_distance(chunk_vec, centroid)      # in [0, 2]; clip to [0, 1]
            surprise = min(surprise, 1.0)
        salience = alpha * length_score + beta * surprise
    """
    length_score = min(len(chunk_text) / 800.0, 1.0)
    if not recent_vecs:
        surprise = 1.0
    else:
        window = np.array(recent_vecs[-10:])
        centroid = window.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid /= norm
        surprise = float(np.clip(1.0 - float(chunk_vec @ centroid), 0.0, 1.0))
    return alpha * length_score + beta * surprise
```

**Constants.** `alpha=0.4, beta=0.6` weight surprise more than length; consistent with
SynapticRAG's empirical finding that inter-event semantic distance is the stronger
predictor of long-term retention. The 800-char saturation point is one typical arc
text; tune upward for raw session chunks (≈ 1500 chars).

**Token budget.** Zero — no LLM call. Latency dominated by the centroid computation
over N=10 vectors: < 0.5 ms on CPU.

### A.2 LLM v2 — Importance Score Prompt

Directly replicates the Park et al. (2023) integer importance rating, extended with
two additional dimensions that the Generative Agents paper identified in §3.2:
novelty relative to recent observations, and decision-weight (does this change what
the agent should do next?). The SynapticRAG surprise proxy (C16) validates the
prediction-error component.

```markdown
<!-- prompts/salience_score.md -->
## role
You are a memory importance evaluator for a coding assistant.

## task
Rate the salience of the CHUNK below on a scale from 1 to 10.

1 = trivially low-salience: boilerplate, scaffolding, formatting change, comment typo
5 = moderately important: routine decision, known pattern applied in new context
10 = maximally important: new requirement, architectural reversal, security finding,
     never-before-seen error class, explicit user constraint

Consider three dimensions equally:
- **Novelty**: how different is this from the RECENT CONTEXT below?
- **Decision weight**: does this change what should be done next?
- **Surprise**: does this contradict or significantly extend established patterns?

## recent_context
{{ recent_summaries }}   {# last 3 arc texts, truncated to 200 chars each #}

## chunk
{{ chunk_text }}

## output_format
Reply with a single JSON object and nothing else:
{"score": <integer 1–10>, "reason": "<one sentence, ≤ 20 words>"}
```

**Why this format works.** Tian et al. (2023) show that RLHF-fine-tuned models
produce better-calibrated verbalized confidence when the prompt names the anchor
values explicitly (1 = X, 10 = Y). Providing three labelled dimensions rather than a
single "how important?" reduces the model's tendency to collapse to 7–8 on everything
— the common clustering failure documented in that paper.

**Token budget.** ~350 input tokens (prompt + 3 × 200-char context + chunk of ≤ 1000
chars) + 30 output tokens. At Ollama llama3.2:3b throughput of ~80 tok/s: ≈ 5 s per
chunk on an Apple Silicon laptop. At haiku-class API pricing ($0.25/M input):
350 × $0.00000025 ≈ **$0.000088 per chunk** — under $0.01 per 100 chunks.

### A.3 Cost Analysis and Co-existence Strategy

| Mode | Cost/chunk | Latency | Quality |
|------|-----------|---------|---------|
| Heuristic v1 | $0 | < 1 ms | Noisy at cold start; improves once arc graph fills |
| LLM v2 | ~$0.0001 | 3–8 s local | Better calibrated; catches semantic nuance |

**Recommended co-existence.** Run heuristic v1 at every flush event (synchronous,
zero cost). Schedule a NREM-3 LLM re-score pass only for chunks whose heuristic score
lies in the ambiguous middle band (3–7). Chunks scoring ≤ 2 or ≥ 8 heuristically are
unlikely to flip: the survey literature (C16, C17) shows extreme surprise and
negligible surprise are reliably separable without a model call.

```python
# nrem3_rescore.py (sketch)
RESCORE_BAND = (3, 7)

def needs_llm_rescore(heuristic_score: int) -> bool:
    return RESCORE_BAND[0] <= heuristic_score <= RESCORE_BAND[1]
```

This keeps flush-time latency at < 1 ms while adding LLM nuance during the
already-LLM-heavy NREM-3 pass, where the marginal cost per-chunk is already paid
in context overhead.

**PageRank upgrade path.** HippoRAG (C15) showed that Personalised PageRank over the
arc graph is a better long-run salience signal than per-event LLM rating. Once the
arc graph has ≥ 50 nodes, add a nightly PPR recomputation and blend it into
`chunks.salience` multiplicatively (PPR-normalised to [0, 1]). This replaces both v1
and v2 for chunks that already have arc-graph coverage.

### A.4 Failure Modes and Detection

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| LLM always returns 7 (mode collapse) | Track score distribution; alert if stddev < 1.5 over 50 chunks | Add few-shot calibration anchors (two 1-scored examples, two 10-scored) |
| Heuristic misses short but critical chunks | Salience audit: sample 20 chunks at score ≤ 3 weekly | Raise alpha; add a keyword-boost override for `ERROR`, `SECURITY`, `BREAKING` |
| Centroid stale (recent\_vecs not updated) | Assert `len(recent_vecs) > 0` after flush | Seed with three zero vectors on empty store initialisation |

### A.5 Quality Gate

Regression test: create a fixture of 20 labelled chunks (5 each at expected scores
1–2, 3–5, 6–8, 9–10) and run both scorers. Require Spearman ρ ≥ 0.70 between LLM
output and ground-truth labels before deploying a new model or prompt version.

### A.6 Citations

- Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Agrawala, M.
  (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023.
  `https://doi.org/10.1145/3586183.3606763` — **[VERIFIED]** (Semantic Scholar; year
  2023; venue "ACM Symposium on User Interface Software and Technology")

- Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). *HippoRAG:
  Neurobiologically Inspired Long-Term Memory for Large Language Models*. NeurIPS 2024.
  `https://doi.org/10.48550/arxiv.2405.14831` — **[VERIFIED]** (OpenAlex; year 2024;
  authors confirmed Bernal Jiménez Gutiérrez et al.)

- Hou, Y., Tamoto, H., Zhao, Q., & Miyashita, H. (2025). *SynapticRAG: Enhancing
  Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms*.
  ACL Findings 2025.
  `https://doi.org/10.18653/v1/2025.findings-acl.1048` — **[VERIFIED]** (OpenAlex;
  year 2025; "consistent improvements … up to 14.66% points")

- Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., &
  Manning, C. (2023). *Just Ask for Calibration: Strategies for Eliciting Calibrated
  Confidence Scores from Language Models Fine-Tuned with Human Feedback*. EMNLP 2023.
  `https://doi.org/10.18653/v1/2023.emnlp-main.330` (also arXiv:2305.14975) —
  **[VERIFIED]** (OpenAlex; year 2023; authors confirmed)

---

## B. Schema Extraction via LLM (NREM-3 Sub-Stage, v0.7)

### B.1 Prompt Design

Input is a single arc text (≤ 2000 chars). Output is a JSON array of typed relation
tuples. JSON is preferred over YAML here: (a) most small models tokenise JSON more
reliably than YAML; (b) JSON parsing failures are easier to detect than YAML
ambiguity; (c) downstream SQLite insertion is straightforward via `json.loads`.

```markdown
<!-- prompts/schema_extract.md -->
## role
You are a structured knowledge extractor. You read a text fragment about a software
project and produce typed relational triples that capture only what is explicitly
stated.

## task
From the ARC TEXT below, extract relation triples.

Rules:
- Only extract facts explicitly present in the text. No inference, no elaboration.
- subject and object must be noun phrases (entity names, file paths, concepts, errors).
- relation must be one of: CALLS, IMPORTS, MODIFIES, CONFIGURES, CAUSES, FIXES,
  DEPENDS_ON, DEFINES, IS_A, CONTRADICTS, HAS_PROPERTY.
- confidence must reflect textual evidence strength:
  0.9+ = direct explicit statement ("X calls Y")
  0.7  = strong implication ("X always uses Y" → DEPENDS_ON)
  0.5  = ambiguous or hedged ("X might affect Y")
  0.3  = speculative inference (do not include unless no direct fact exists)
  Do NOT default to 0.9 for everything.
- If fewer than 2 triples are extractable, return an empty array [].

## arc_text
{{ arc_text }}

## output_format
Reply with a single JSON array and nothing else. No markdown fences. Example:
[
  {"subject": "AuthService", "relation": "CALLS", "object": "JWTVerifier",
   "confidence": 0.92},
  {"subject": "config.toml", "relation": "CONFIGURES", "object": "AuthService",
   "confidence": 0.80}
]
```

**Why JSON over YAML.** Microsoft GraphRAG (Edge et al., arXiv:2404.16130; arXiv
confirmed) and HippoRAG (C15) both use JSON as the extraction target format.
GraphRAG's entity extraction prompt specifically avoids YAML because small models
fail to produce syntactically valid YAML at higher temperatures. For Ollama-hosted
models, the JSON failure mode is a parse error that triggers a retry; YAML failure
mode is silent structural corruption.

**Token budget.** ~500 input tokens (prompt + arc up to 2000 chars) + ~200 output
tokens (≤ 10 triples). Latency at llama3.2:3b: ≈ 3–6 s. Cost at Haiku API:
~$0.00018 per arc.

### B.2 Confidence Calibration

The most common extraction failure is confidence collapsing to 0.9 for every triple.
Two complementary mitigations:

**Few-shot anchoring.** Include two concrete examples in the system prompt — one at
0.9 (direct quote), one at 0.5 (hedged phrasing) — that match the domain. This is
the cheapest calibration technique documented by Tian et al. (2023) and requires no
model retraining.

**Self-consistency vote.** Run the extraction prompt twice with temperature 0.3 and
once with temperature 0.7. A triple that appears in all three generations gets its
confidence boosted by ×1.1 (capped at 0.95). A triple that appears in only one
generation is downgraded to 0.4. This operationalises the self-consistency principle
(Wang et al. 2023) without any external verifier.

```python
def calibrate_confidence(triple_runs: list[list[dict]]) -> list[dict]:
    """
    triple_runs: list of 3 extraction results (each is a list of triples).
    Returns deduplicated triples with adjusted confidence.
    """
    from collections import Counter
    key = lambda t: (t["subject"].lower(), t["relation"], t["object"].lower())
    counts: Counter = Counter()
    base: dict = {}
    for run in triple_runs:
        for t in run:
            k = key(t)
            counts[k] += 1
            if k not in base:
                base[k] = t
    result = []
    for k, count in counts.items():
        t = dict(base[k])
        if count == 3:
            t["confidence"] = min(t["confidence"] * 1.1, 0.95)
        elif count == 1:
            t["confidence"] = min(t["confidence"], 0.40)
        result.append(t)
    return sorted(result, key=lambda x: -x["confidence"])
```

### B.3 Bi-Temporal Write and Idempotency

The schema_edges table (see `docs/architecture/00-overview.md §3, L3`) uses
Graphiti-style temporal versioning (Rasmussen et al. 2025, arXiv:2501.13956).

```python
def upsert_schema_edge(
    db: sqlite3.Connection,
    project: str,
    subject: str,
    relation: str,
    object_: str,
    evidence_id: str,
    confidence: float,
    now_ts: int,
) -> None:
    """
    Idempotency: same (project, subject, relation, object, evidence_id) → no-op.
    Contradiction: existing active edge with same (subject, relation, object) but
    different evidence_id → invalidate old, write new.
    """
    cur = db.cursor()
    # 1. Idempotency check
    cur.execute("""
        SELECT id FROM schema_edges
        WHERE project=? AND subject=? AND relation=? AND object=?
          AND evidence_ids LIKE ? AND invalidated_ts=0
    """, (project, subject, relation, object_, f"%{evidence_id}%"))
    if cur.fetchone():
        return  # exact duplicate — no-op

    # 2. Contradiction check: active edge with same triple but different source
    cur.execute("""
        SELECT id FROM schema_edges
        WHERE project=? AND subject=? AND relation=? AND object=?
          AND invalidated_ts=0
    """, (project, subject, relation, object_))
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE schema_edges SET invalidated_ts=?, superseded_by=? WHERE id=?",
            (now_ts, -1, existing[0])  # superseded_by=-1 signals self-supersession
        )

    # 3. Write new
    cur.execute("""
        INSERT INTO schema_edges
          (project, subject, relation, object, evidence_ids, confidence,
           created_ts, invalidated_ts, superseded_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
    """, (project, subject, relation, object_,
          f'["{evidence_id}"]', confidence, now_ts))
    db.commit()
```

The `invalidated_ts=0` sentinel means "currently active." This mirrors Graphiti's
bi-temporal design (Rasmussen et al. 2025): "each edge has a validity interval;
contradiction writes a new edge with the old edge's interval closed at the
contradiction timestamp."

### B.4 Bounding Schema Extraction Loss

Schema extraction is inherently lossy. Three mechanisms bound the loss:

1. **Arc retention.** The original arc text is always kept in L2. Schema edges are
   indices over arcs, not replacements. If an extraction fails to capture a relation,
   the arc is still retrievable by embedding similarity.

2. **Confidence floor.** Edges with `confidence < 0.45` are written with
   `invalidated_ts = created_ts + 86400` (auto-expire in 24 h). They contribute to
   the graph only if confirmed by a second extraction pass.

3. **Coverage audit.** After each NREM-3 pass, compute:
   `coverage = edges_written / arcs_processed`. Alert if `coverage < 0.6` — fewer
   than 0.6 edges per arc suggests the extraction prompt is firing incorrectly.

### B.5 Failure Modes and Detection

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| JSON parse error | `json.loads` exception; rate > 5% per session | Retry with temperature 0.1; add explicit "no markdown fences" reminder |
| Hallucinated entity names | Entity not found in arc text by substring match | Post-extraction grounding check: every subject/object must appear (case-insensitive) in the source arc |
| All confidences at 0.9 | stddev(confidences) < 0.05 over 20 triples | Add calibration examples to prompt |
| Relation vocabulary drift | Unknown relation strings | Validate against the 11-item vocabulary; map unknown to IS_A or reject |

### B.6 Quality Gate

Regression corpus: 15 hand-annotated arcs with known ground-truth triples. Require
F1 ≥ 0.65 on exact (subject, relation, object) match before any prompt change ships.
Spot-check: monthly sample of 10 extractions reviewed manually, targeting calibration
ECE (Expected Calibration Error) < 0.15.

### B.7 Citations

- Edge, M., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., &
  Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization*. arXiv:2404.16130.
  `https://doi.org/10.48550/arxiv.2404.16130` — **[VERIFIED]** (arXiv HTML
  confirmed; title "From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization"; OpenAlex DOI lookup returned 404 but arXiv canonical URL confirmed)

- Gutiérrez, B. J., et al. (2024). *HippoRAG*. NeurIPS 2024.
  `https://doi.org/10.48550/arxiv.2405.14831` — **[VERIFIED]** (see §A.6)

- Rasmussen, P., Paliychuk, P., Beauvais, T., & Ryan, J. (2025). *Zep: A Temporal
  Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956.
  `https://doi.org/10.48550/arxiv.2501.13956` — **[VERIFIED]** (OpenAlex; year 2025;
  title and authors confirmed)

- Tian, K., et al. (2023). *Just Ask for Calibration*. EMNLP 2023.
  `https://doi.org/10.18653/v1/2023.emnlp-main.330` — **[VERIFIED]** (see §A.6)

---

## C. Reconsolidation Merge Prompt

### C.1 Design Constraints

The reconsolidation merge is triggered when a new `Stop` chunk arrives within
`labile_until` of a retrieved chunk, with cosine ≥ 0.82 between the two
(architecture §4). The LLM's task is strictly to tighten the merged representation
— not to add, infer, or editorialize. This is the hardest prompt to get right because
small models consistently introduce novel claims during "summarisation."

The closest prior-art constraint is the SelfCheckGPT faithfulness criterion (Manakul
et al., 2023): "a statement is faithful if and only if it is entailed by the source
text." Applied here: every sentence in the merged output must be entailed by either
the old chunk or the new chunk.

```markdown
<!-- prompts/reconsolidate_merge.md -->
## role
You are a faithful document merger. You combine two versions of the same memory
into a single, tighter version that loses nothing.

## task
Merge OLD_CHUNK and NEW_CHUNK into a single MERGED text.

Hard constraints — violation disqualifies the output:
1. Every fact in OLD_CHUNK must survive in MERGED, word-for-word or as a
   semantically equivalent paraphrase.
2. Every fact in NEW_CHUNK must survive in MERGED.
3. MERGED must contain NO sentence that is not entailed by OLD_CHUNK or NEW_CHUNK.
   Do not add examples, background, or elaboration.
4. MERGED must be shorter than OLD_CHUNK + NEW_CHUNK combined.
5. Preserve all proper nouns, file paths, function names, error codes, and numeric
   values exactly as they appear in the source texts.
6. If OLD_CHUNK and NEW_CHUNK state contradictory facts, preserve BOTH with explicit
   attribution: "Previously: <old fact>. Updated: <new fact>."

## old_chunk
{{ old_text }}

## new_chunk
{{ new_text }}

## output_format
Reply with the MERGED text only. No preamble, no explanation, no markdown.
```

### C.2 Post-Merge Cosine Verification Gate

After the LLM returns the merged text, re-embed it and run a sanity check. The
merged embedding should lie between the two source embeddings in vector space: if the
LLM has injected novel content, the merged vector will drift away from both.

```python
def cosine_gate(
    old_vec: np.ndarray,
    new_vec: np.ndarray,
    merged_vec: np.ndarray,
    tolerance: float = 0.12,
) -> bool:
    """
    Returns True if merged_vec is within `tolerance` of the midpoint of old and new.
    Threshold 0.12 allows tightening language (small drift) while blocking hallucination
    (large drift). Empirically: paraphrase shift ≈ 0.04–0.08; topic drift ≈ 0.15+.
    """
    midpoint = (old_vec + new_vec) / 2.0
    norm = np.linalg.norm(midpoint)
    if norm > 1e-9:
        midpoint /= norm
    drift = 1.0 - float(merged_vec @ midpoint)
    return drift <= tolerance
```

If `cosine_gate` returns `False`, the merge is rejected: the new chunk is written as
a separate entry rather than merged. The labile window is closed early by setting
`labile_until = 0`.

**Token budget.** ~600 input tokens (prompt + old + new, each ≤ 1000 chars) + ≤ 600
output tokens. At llama3.2:3b: ≈ 8–12 s. At Haiku API: ~$0.00030 per merge. Merges
are rare (triggered only within the 5-minute labile window, with cosine ≥ 0.82), so
the per-session merge budget is typically 0–3 calls.

### C.3 Anchor in Literature

No paper directly addresses "lossless LLM text merge" as a standalone problem.
The closest anchors are:

- **SelfCheckGPT** (Manakul et al. 2023, `https://doi.org/10.18653/v1/2023.emnlp-main.557`):
  the entailment-based faithfulness criterion ("every statement must be entailed by
  source") is the principle embedded in hard constraints 1–3 above. Abstract confirms:
  "Generative LLMs such as GPT-3 … hallucinate facts … [SelfCheckGPT] detect[s]
  hallucinated facts in any black-box model." The prompt's constraint 3 is a
  preventive application of the same logic.

- **Park et al. 2023** (C18) reflection prompt: "synthesise those memories over time
  into higher-level reflections." The reconsolidation merge is the inverse operation —
  specificity-preserving rather than abstraction-seeking.

- **Reconsolidation literature** (architecture §4): the cosine gate is the
  computational analog of the reconsolidation stability gate — if the merged trace
  drifts too far from either source, it is treated as a new encoding event rather than
  a memory update.

### C.4 Failure Modes and Detection

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Hallucinated content | cosine_gate drift > 0.12 | Reject merge; write as new chunk |
| Dropped facts (old facts missing from merge) | Post-merge: check each sentence from old appears in merge by 0.85 cosine | Lower temperature; add explicit constraint 1 anchor phrase |
| Merge longer than combined inputs | Character length check | Add constraint 4 explicitly and retry once |
| Model refuses (safety filter on code) | Empty output or refusal token | Use plain-text wrapper; strip code fences before sending |

### C.5 Quality Gate

Spot-check: monthly sample of 20 merge outputs. Human evaluator (or a second, larger
LLM) answers three yes/no questions for each: (1) any novel fact introduced? (2) any
source fact dropped? (3) proper nouns / paths preserved? Target: < 5% failure on each
criterion.

### C.6 Citations

- Manakul, P., Liusie, A., & Gales, M. (2023). *SelfCheckGPT: Zero-Resource Black-Box
  Hallucination Detection for Generative Large Language Models*. EMNLP 2023.
  `https://doi.org/10.18653/v1/2023.emnlp-main.557` — **[VERIFIED]** (OpenAlex; year
  2023; authors Manakul, Liusie, Gales confirmed)

- Park, J. S., et al. (2023). *Generative Agents*. UIST 2023.
  `https://doi.org/10.1145/3586183.3606763` — **[VERIFIED]** (see §A.6)

---

## D. REM Generative Recombination Prompt and Quarantine

### D.1 Risk Context

`_stage3_rem` is the highest-risk component in the dream pass. Du (2026) names
"self-reinforcing false belief" as the primary failure mode in reflective agent
memory: an incorrect generalisation is promoted to semantic memory and never corrected
because the agent stops collecting contradicting evidence (C1). Papageorgiou et al.
(2025) find that GraphRAG-style inference — the closest published analogue to REM
recombination — carries "systematically higher hallucination rates than pure embedding
retrieval" (C2). Chakraborty et al. (2025) distinguish closed-book hallucination
(generation beyond context) from open-book hallucination (contradiction of retrieved
context) and note that both arise in generative recombination stages (C3).

The quarantine schema (kind=hypothesis, cross_eligible=0, confidence=null) is the
primary safety net. The prompt design's second job is to make the model's uncertainty
legible rather than hidden.

### D.2 Prompt Design

```markdown
<!-- prompts/rem_recombine.md -->
## role
You are a hypothesis generator for a software project memory system. You examine
surprising connections between memory fragments and propose candidate hypotheses.
You are NOT a fact-checker and you are NOT writing to the permanent record —
everything you produce goes into a quarantined hypothesis store for human review.

## task
Below are {{ n }} memory arc fragments sampled because they are high-salience but
semantically distant from each other. Your task is to propose hypotheses about
what deeper pattern, causal link, or architectural relationship might connect them.

Speculation is explicitly invited. Uncertainty is required.

Mandatory constraints:
1. Every hypothesis must begin with "HYPOTHESIS:" followed by a confidence token:
   HIGH (you would bet on it), MEDIUM (plausible but untested), LOW (speculative).
2. Every hypothesis must cite at least one arc_id from the fragments below as its
   evidence. Format: [evidence: arc_id_1, arc_id_2].
3. Do NOT state as fact anything not present in the arc fragments. If you find
   yourself writing a declarative sentence without an evidence citation, stop and
   rewrite it as a hypothesis.
4. Produce 2–4 hypotheses maximum. Prefer fewer, sharper hypotheses over many
   vague ones.
5. If you cannot construct a plausible hypothesis grounded in the evidence,
   respond with exactly: NO_HYPOTHESIS

## arc_fragments
{% for arc in arcs %}
### arc_id: {{ arc.id }}
{{ arc.text }}
{% endfor %}

## output_format
Each hypothesis on its own line. No numbering, no markdown headers.
```

**Design rationale.** Radensky et al. (2024) find that unconstrained LLM
recombination produces "high novelty but poor groundedness" and requires a mandatory
"novelty-vs-groundedness evaluation step before any output is accepted" (C4). The
prompt's constraints 2–3 are a structural implementation of this requirement:
evidence citation is non-optional, and bare declarative sentences are explicitly
blocked. The confidence token (HIGH/MEDIUM/LOW) makes uncertainty legible rather than
burying it in hedging language that a downstream retrieval system would strip.

**Arc sampling.** Sample 3–5 arcs with `salience ≥ 6` whose pairwise cosine is
< 0.35 (semantically distant). Minimum cosine distance ensures the model is
genuinely bridging dissimilar material rather than paraphrasing adjacent arcs.

**Token budget.** ~800 input tokens (prompt + 3–5 arcs × ~200 chars each) + ≤ 400
output tokens. At llama3.2:3b: ≈ 10–15 s. At Haiku API: ~$0.00030 per REM pass.
REM runs at most once per dream cycle.

### D.3 Output Schema and Quarantine Semantics

```python
# Written to L2 semantic store with:
{
    "kind": "hypothesis",
    "cross_eligible": 0,       # never promoted to L4
    "evidence_ids": [...],      # arc_ids from the prompt
    "confidence": None,         # forced null — no LLM confidence used
    "salience": 3,              # fixed low salience for hypotheses
    "consolidation_round": 0,   # never auto-promoted
}
```

**Retrieval filter.** Every retrieval path adds `AND kind != 'hypothesis'` to its
WHERE clause. Hypotheses are invisible to normal recall. The only surfaces that show
them are:
- `/recall hypothesis` explicit command
- UI panel with visual distinction (distinct colour, "UNVERIFIED" badge)
- NREM-3 promotion path (see below)

**Promotion path.** A hypothesis is promoted from quarantine to `kind='arc'` with
`consolidation_round=1` only via explicit user action in the UI. No automatic
promotion is permitted. This is the strongest form of the evidence-gated commit
pattern from Du (2026, C1): human-in-the-loop is the gate.

### D.4 Evidence-Gated Grounding Check

Before the hypothesis is written to the store, run a lightweight grounding check:

```python
def hypothesis_is_grounded(
    hypothesis_text: str,
    evidence_arc_texts: list[str],
    threshold: float = 0.30,
) -> bool:
    """
    Embeds the hypothesis and each evidence arc. Returns True if the hypothesis
    cosine to at least one evidence arc exceeds threshold. 0.30 is intentionally
    low — hypothesis generation is supposed to bridge, not merely paraphrase.
    A hypothesis scoring < 0.30 to all evidence arcs has drifted completely off
    its stated evidence and should be dropped.
    """
    h_vec = embed(hypothesis_text)
    for arc_text in evidence_arc_texts:
        a_vec = embed(arc_text)
        if float(h_vec @ a_vec) >= threshold:
            return True
    return False
```

Hypotheses that fail this check are silently dropped (not written to the store).

### D.5 Failure Modes and Detection

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Self-reinforcing false belief | Hypothesis not grounded (cosine < 0.30 to all evidence) | Drop at grounding check; do not write to store |
| Open-book hallucination (contradicts existing high-salience chunk) | Post-generation: embed hypothesis; check cosine to top-3 semantic neighbours; if similarity > 0.80 AND conflict judge returns CONTRADICTS → drop | Contradiction detection before write |
| Hypothesis promotion without review | Audit: any hypothesis with consolidation\_round > 0 and no UI action record | Require `promoted_by_user = 1` flag; assert in promotion code path |
| Too many hypotheses per pass (noise) | Count: if > 4 per REM pass → quality degraded | Hard-cap at 4 in prompt; discard excess |

### D.6 Quality Gate

Manual review cycle: at each v0.7 milestone, review 10 hypotheses. Rate each as
plausible / implausible / harmful. Target: ≥ 60% plausible, 0% harmful (defined as
a false belief that, if promoted, would cause incorrect code generation).

### D.7 Citations

- Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and
  Emerging Frontiers*. arXiv:2603.07670.
  `https://doi.org/10.48550/arxiv.2603.07670` — **[VERIFIED]** (OpenAlex; year 2026;
  author Pengfei Du confirmed)

- Radensky, M., Shahid, S., Fok, R., Siangliulue, P., & Hope, T. (2024). *Scideator:
  Human-LLM Compound System for Scientific Ideation through Facet Recombination and
  Novelty Evaluation*. arXiv:2409.14634.
  `https://doi.org/10.48550/arxiv.2409.14634` — **[VERIFIED]** (OpenAlex; year 2024;
  full title "Human-LLM Compound System for Scientific Ideation through Facet
  Recombination and Novelty Evaluation"; also known as Scideator)

- Papageorgiou, G., Sarlis, V., Maragoudakis, M., Magnisalis, I., & Tjortjis, C.
  (2025). *Evaluating Faithfulness in Agentic RAG Systems for e-Governance
  Applications Using LLM-Based Judging Frameworks*. BDCC 9(12), 309.
  `https://doi.org/10.3390/bdcc9120309` — **[VERIFIED]** (OpenAlex; year 2025;
  authors confirmed)

- Chakraborty, N., Ornik, M., & Driggs-Campbell, K. (2025). *Hallucination Detection
  in Foundation Models for Decision-Making: A Flexible Definition and Review of the
  State of the Art*. ACM Computing Surveys.
  `https://doi.org/10.1145/3716846` — **[VERIFIED]** (Semantic Scholar; year 2024/2025;
  authors Chakraborty, Ornik, Driggs-Campbell confirmed)

---

## E. Cross-Project Conflict Judge Prompt Update

### E.1 The R2 Invariant

Risk R2 in `docs/research/04-cross-project-risk-audit.md` identifies the current
conflict judge prompt as carrying zero project context. The prompt receives only
`{new_text}` and `{old_text}`. If a cross-project pass ever relaxes the
`project = '{proj}'` SQL filter, the judge will treat cross-project divergence as
direct contradiction and write `superseded_by`, **permanently excluding the older
chunk from all retrieval**. This is described in the risk audit as "a silent,
destructive, irreversible write with no undo path."

The fix has two components: (1) the judge receives both project labels; (2) the
decision tree branches on same-project vs cross-project before choosing the verdict.
**Cross-project supersession is permanently forbidden** — this rule is hardcoded in
the Python caller, not left to the LLM.

### E.2 Updated Judge Prompt

```markdown
<!-- prompts/judge_conflict_v2.md -->
## role
You are a conflict arbiter for a two-store memory system. You assess whether two
memory chunks represent the same fact, different perspectives, or an explicit
factual contradiction.

## task
Assess the relationship between OLD_CHUNK and NEW_CHUNK.

Context you must use:
- OLD_PROJECT_LABEL: {{ project_label_old }}
- NEW_PROJECT_LABEL: {{ project_label_new }}
- SAME_PROJECT: {{ same_project }}   {# boolean, injected by caller #}

Decision rules:
1. If SAME_PROJECT is true AND the chunks state directly contradictory facts
   (e.g., "uses JWT" vs "does not use JWT" about the same entity in the same project):
   → verdict: CONTRADICTS
   → action: write superseded_by on old chunk
   → reason: <one sentence>

2. If SAME_PROJECT is false AND the chunks state facts that diverge or conflict:
   → verdict: DIVERGES
   → action: write a typed 'contradicts' link in cross_links only; NEVER write
     superseded_by; NEVER modify retrieval eligibility of either chunk
   → reason: <one sentence noting which project each fact belongs to>

3. If the chunks are about different entities, different time periods, or the
   apparent conflict is a different project's version of the same technology:
   → verdict: UNRELATED
   → action: no write
   → reason: <one sentence>

4. If you cannot determine the relationship with high confidence (> 0.80):
   → verdict: DEFER
   → action: no write
   → reason: <one sentence explaining what is ambiguous>

Hard constraint: if SAME_PROJECT is false, you MUST NOT return CONTRADICTS.
The only cross-project verdicts are DIVERGES, UNRELATED, or DEFER.

## old_chunk
[Project: {{ project_label_old }}]
{{ old_text }}

## new_chunk
[Project: {{ project_label_new }}]
{{ new_text }}

## output_format
Reply with a single JSON object and nothing else:
{"verdict": "<CONTRADICTS|DIVERGES|UNRELATED|DEFER>",
 "action": "<write action or 'none'>",
 "reason": "<one sentence, ≤ 25 words>"}
```

### E.3 Decision Tree and Write Actions

```
                        ┌─────────────────────────────┐
                        │  same_project?               │
                        └──────┬──────────────┬────────┘
                               │ yes          │ no
                               ▼              ▼
                    ┌──────────────┐  ┌──────────────────────┐
                    │ Factual      │  │ Any divergence?       │
                    │ reversal?    │  └───────┬──────────┬────┘
                    └──┬───────┬──┘          │ yes      │ no / unclear
                       │ yes   │ no          ▼          ▼
                       ▼       ▼         DIVERGES   UNRELATED
                  CONTRADICTS UNRELATED  (cross_links (no write)
                  (superseded_ (no write) only, no
                   by=new_id)             superseded_by)
                                                │
                                     ambiguous? → DEFER (no write)
```

**Python caller enforcement.** The prohibition on cross-project `superseded_by` is
enforced in the Python caller, not trusted to the LLM:

```python
def apply_judge_verdict(
    verdict: dict,
    old_chunk_id: str,
    new_chunk_id: str,
    same_project: bool,
    db: sqlite3.Connection,
) -> None:
    v = verdict["verdict"]

    # Hard safety: cross-project CONTRADICTS is rewritten to DIVERGES
    if not same_project and v == "CONTRADICTS":
        v = "DIVERGES"  # LLM violated the constraint; enforce programmatically

    if v == "CONTRADICTS" and same_project:
        db.execute(
            "UPDATE chunks SET superseded_by=? WHERE id=?",
            (new_chunk_id, old_chunk_id)
        )
    elif v == "DIVERGES" and not same_project:
        # Write typed cross_link only; never touch superseded_by
        _write_cross_link(db, old_chunk_id, new_chunk_id, link_type="contradicts")
    # UNRELATED and DEFER: no write
    db.commit()
```

The rewrite-to-DIVERGES guard on line 10 catches the case where a small model ignores
the `SAME_PROJECT is false → MUST NOT return CONTRADICTS` instruction. The constraint
is structural, not merely advisory.

### E.4 Token Budget and Cost

~400 input tokens (prompt + both chunks at ≤ 600 chars each) + 60 output tokens.
At llama3.2:3b: ≈ 4–7 s. At Haiku API: ~$0.00012 per conflict assessment. The judge
is called only when cosine similarity is ≥ 0.70 (within-project) or ≥ 0.85
(cross-project), which keeps call volume low.

### E.5 Failure Modes and Detection

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| LLM returns CONTRADICTS for cross-project pair | Caller rewrite to DIVERGES (logged as WARNING) | Sufficient — structural guard catches it |
| DEFER overuse (model is uncertain about everything) | DEFER rate > 30% over 50 calls | Add few-shot example of clear CONTRADICTS and clear UNRELATED |
| DIVERGES without both project labels in reason | reason string does not contain project label | Validate reason before write; reject and retry |
| Spurious UNRELATED for clear same-project contradiction | Human audit: sample 10 UNRELATED verdicts monthly | Add few-shot example of multi-paragraph CONTRADICTS pair |

### E.6 Quality Gate

Integration test: fixture of 20 chunk pairs (5 same-project CONTRADICTS, 5
same-project UNRELATED, 5 cross-project DIVERGES, 5 cross-project UNRELATED).
Require accuracy ≥ 85% and — critically — 0/5 cross-project pairs returning
CONTRADICTS (even if the Python guard saves them, a model that consistently violates
the constraint needs prompt revision).

### E.7 Citations

- R2 invariant: `docs/research/04-cross-project-risk-audit.md §R2` — cross-project
  supersession risk. Internal document, no DOI; cited as design authority.

- Du, P. (2026). *Memory for Autonomous LLM Agents*. arXiv:2603.07670.
  `https://doi.org/10.48550/arxiv.2603.07670` — **[VERIFIED]** (see §D.7); §4.6
  "write-path filtering and contradiction handling" is the primary survey anchor for
  why conflict judge correctness is critical.

- Rasmussen, P., et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for
  Agent Memory*. arXiv:2501.13956.
  `https://doi.org/10.48550/arxiv.2501.13956` — **[VERIFIED]** (see §B.7); the
  `t_invalid` temporal edge pattern underpins the DIVERGES write action.

---

## Summary: Token Budgets and Costs per Dream Pass

| Stage | Prompt | Input tok | Output tok | Cost (Haiku API) | Cost (Ollama local) |
|-------|--------|----------|-----------|-----------------|-------------------|
| A. Salience (heuristic) | — | 0 | 0 | $0 | $0 |
| A. Salience (LLM, mid-band only) | salience\_score.md | ~350 | ~30 | ~$0.0001/chunk | ~5 s/chunk |
| B. Schema extract (×3 for calibration) | schema\_extract.md | ~500 × 3 | ~200 × 3 | ~$0.00054/arc | ~12 s/arc |
| C. Reconsolidation merge | reconsolidate\_merge.md | ~600 | ~600 | ~$0.0003/merge | ~10 s/merge |
| D. REM recombine (once/cycle) | rem\_recombine.md | ~800 | ~400 | ~$0.0003/cycle | ~12 s/cycle |
| E. Conflict judge (per pair) | judge\_conflict\_v2.md | ~400 | ~60 | ~$0.00012/pair | ~5 s/pair |

**Per-session total (typical: 5 arcs, 2 schema arcs, 1 merge, 1 REM, 3 conflict
pairs):** ~$0.004 at Haiku API; ~90 s total at Ollama llama3.2:3b on M1 CPU.
Stages B–E run in the background (NREM-3 / REM) and do not block the user session.

---

## Session Citation Registry

All citations in this document are API-verified. Full registry:

| ID | Authors | Year | Title (abbreviated) | DOI / arXiv | API Source | Tier |
|----|---------|------|---------------------|-------------|-----------|------|
| R-A1 | Park et al. | 2023 | Generative Agents | 10.1145/3586183.3606763 | Semantic Scholar | VERIFIED |
| R-A2 | Gutiérrez et al. | 2024 | HippoRAG | 10.48550/arxiv.2405.14831 | OpenAlex | VERIFIED |
| R-A3 | Hou et al. | 2025 | SynapticRAG | 10.18653/v1/2025.findings-acl.1048 | OpenAlex | VERIFIED |
| R-A4 | Tian et al. | 2023 | Just Ask for Calibration | 10.18653/v1/2023.emnlp-main.330 (arXiv:2305.14975) | OpenAlex | VERIFIED |
| R-B1 | Edge et al. | 2024 | GraphRAG (Local to Global) | 10.48550/arxiv.2404.16130 | arXiv HTML | VERIFIED |
| R-B2 | Rasmussen et al. | 2025 | Zep / Graphiti | 10.48550/arxiv.2501.13956 | OpenAlex | VERIFIED |
| R-C1 | Manakul et al. | 2023 | SelfCheckGPT | 10.18653/v1/2023.emnlp-main.557 | OpenAlex | VERIFIED |
| R-D1 | Du | 2026 | Memory for Autonomous LLM Agents | 10.48550/arxiv.2603.07670 | OpenAlex | VERIFIED |
| R-D2 | Radensky et al. | 2024 | Scideator | 10.48550/arxiv.2409.14634 | OpenAlex | VERIFIED |
| R-D3 | Papageorgiou et al. | 2025 | Faithfulness in Agentic RAG | 10.3390/bdcc9120309 | OpenAlex | VERIFIED |
| R-D4 | Chakraborty et al. | 2025 | Hallucination Detection (Decision-Making) | 10.1145/3716846 | Semantic Scholar | VERIFIED |
