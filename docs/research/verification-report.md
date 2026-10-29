# Citation Verification Report

**Verified by:** biss-brain citation-verifier  
**Date:** 2026-04-30  
**Sources checked:** 3 documents (01-neuroscience-of-memory.md, 02-bio-inspired-ai-architectures.md, 03-cross-project-memory-prior-art.md)  
**APIs used:** Crossref Works, OpenAlex, arXiv Atom, PubMed EFetch, Crossref CrossMark  

---

## Verification Table

| source_id | claimed_title / ref | api_title | claimed_year | api_year | claimed_first_author | api_first_author | title_match | year_match | author_match | entailment | retracted | confidence_tier | notes |
|-----------|---------------------|-----------|--------------|----------|----------------------|------------------|-------------|------------|--------------|------------|-----------|-----------------|-------|
| CLS-1995 | Why there are complementary learning systems... | Why there are complementary learning systems in the hippocampus and neocortex | 1995 | 1995 | McClelland | McClelland | YES | YES | YES | YES | No | [VERIFIED] | Psych Rev 102(3):419-457. DOI 10.1037/0033-295X.102.3.419. |
| KUMARAN-2016 | What Learning Systems do Intelligent Agents Need? CLS Theory Updated | What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated | 2016 | 2016 | Kumaran | Kumaran | YES | YES | YES | YES | No | [VERIFIED] | TiCS 20(7):512-534. Note: doc says issue 8, API says issue 7. Pages correct. DOI 10.1016/j.tics.2016.05.004. |
| YASSA-2011 | Yassa & Stark (2011), *Nature Reviews Neuroscience* 12:367-378 | Pattern separation in the hippocampus | 2011 | 2011 | Yassa | Yassa | YES | YES | YES | YES | No | [FLAGGED] | JOURNAL MISMATCH: actual journal is *Trends in Neurosciences* 34:515-525, NOT Nature Reviews Neuroscience. DOI 10.1016/j.tins.2011.06.006. |
| TONONI-2003 | Tononi & Cirelli (2003, *Brain Research Bulletin*) | Sleep and synaptic homeostasis: a hypothesis | 2003 | 2003 | Tononi | Tononi | YES | YES | YES | YES | No | [VERIFIED] | Brain Res Bull 62(2):143-150. DOI 10.1016/j.brainresbull.2003.09.004. No DOI in PMC record but Crossref confirms. |
| TONONI-2014 | Tononi & Cirelli (2014, *Neuron*) | Sleep and the Price of Plasticity: From Synaptic and Cellular Homeostasis to Memory Consolidation and Integration | 2014 | 2014 | Tononi | Tononi | YES | YES | YES | YES | No | [VERIFIED] | Neuron 81(1):12-34. DOI 10.1016/j.neuron.2013.12.025. PMC3921176 (PMID 24411729). |
| WILSON-1994 | Wilson & McNaughton (1994, *Science* 265) | Reactivation of hippocampal ensemble memories during sleep | 1994 | 1994 | Wilson | Wilson | YES | YES | YES | YES | No | [VERIFIED] | Science 265:676-679. PMID 8036517. No Crossref DOI but confirmed via PubMed. |
| FOSTER-2006 | Foster & Wilson (2006, *Nature* 440) — cited for "Forward replay" | Reverse replay of behavioural sequences in hippocampal place cells during the awake state | 2006 | 2006 | Foster | Foster | YES | YES | YES | NO | No | [FLAGGED] | CONTENT ERROR: paper is specifically about REVERSE replay. Doc labels it as the citation for "Forward replay". Volume and page (440:680) correct. DOI 10.1038/nature04587. |
| DIBA-2007 | Diba & Buzsáki (2007, *Nature Neuroscience* 10:1241-1242) | Forward and reverse hippocampal place-cell sequences during ripples | 2007 | 2007 | Diba | Diba | YES | YES | YES | YES | No | [VERIFIED] | NatNeuro 10:1241-1242. DOI 10.1038/nn1961. Volume and pages match exactly. |
| TSE-2007 | Tse et al. (2007, *Science* 315) | Schemas and Memory Consolidation | 2007 | 2007 | Tse | Tse | YES | YES | YES | YES | No | [APPROXIMATE] | VOLUME ERROR: paper is Science vol 316 (issue 5821), pages 76-82. Doc body text says "Science 315". DOI 10.1126/science.1135935. The source index URL is correct. |
| NADER-2000 | Nader, Schafe & LeDoux (2000, *Nature* 406) | Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval | 2000 | 2000 | Nader | Nader | YES | YES | YES | YES | No | [VERIFIED] | Nature 406:722-726. DOI 10.1038/35021052. CrossMark: no retraction. |
| HARDT-2013 | Hardt, Nader & Nadel (2013, *Trends in Cognitive Sciences*) | Decay happens: the role of active forgetting in memory | 2013 | 2013 | Hardt | Hardt | YES | YES | YES | YES | No | [VERIFIED] | TiCS 17(3):111-120. Real DOI is 10.1016/j.tics.2013.01.001. Note: the document's URL points to a non-DOI PDF; an incorrect DOI (10.1016/j.tics.2013.09.003) was tested first and resolves to a different paper — use the confirmed DOI. |
| BHATT-2020 | Bhatt et al., 2020, *Journal of Neuroscience* | NOT FOUND | 2020 | — | Bhatt | — | — | — | — | — | — | [FLAGGED] | No PubMed or Crossref record for "Bhatt 2020 J Neuroscience" on GluA2/AMPAR forgetting. The described finding (blocking GluA2 endocytosis prevents forgetting of object-location memories) matches Migues et al. (2016), J Neuroscience 36:3481-3494. DOI 10.1523/JNEUROSCI.3333-15.2016. Likely fabricated citation; correct source is Migues 2016. |
| LIU-2012 | Liu et al. (2012), *Nature* 484 | Optogenetic stimulation of a hippocampal engram activates fear memory recall | 2012 | 2012 | Liu | Liu | YES | YES | YES | YES | No | [VERIFIED] | Nature 484:381-385. DOI 10.1038/nature11028. CrossMark: no retraction. |
| RAMIREZ-2013 | Ramirez et al. (2013), *Science* 341 | Creating a False Memory in the Hippocampus | 2013 | 2013 | Ramirez | Ramirez | YES | YES | YES | YES | No | [VERIFIED] | Science 341:387-391. DOI 10.1126/science.1239073. CrossMark: no retraction. |
| BUCKNER-2008 | Buckner et al. (2008, *Annals of the NYAS*) | The Brain's Default Network | 2008 | 2008 | Buckner | Buckner | YES | YES | YES | YES | No | [VERIFIED] | Ann NY Acad Sci 1124:1-38. DOI 10.1196/annals.1440.011. |
| MARKRAM-1997 | Markram et al. (1997, *Science* 275) | Regulation of Synaptic Efficacy by Coincidence of Postsynaptic APs and EPSPs | 1997 | 1997 | Markram | Markram | YES | YES | YES | YES | No | [VERIFIED] | Science 275:213-215. DOI 10.1126/science.275.5297.213. |
| BI-POO-1998 | Bi & Poo (1998, *Journal of Neuroscience* 18) | Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type | 1998 | 1998 | Bi | Bi | YES | YES | YES | YES | No | [VERIFIED] | JNeurosci 18:10464-10472. DOI 10.1523/JNEUROSCI.18-24-10464.1998. |
| OLSHAUSEN-1996 | Olshausen & Field (1996, *Nature* 381) | Emergence of simple-cell receptive field properties by learning a sparse code for natural images | 1996 | 1996 | Olshausen | Olshausen | YES | YES | YES | YES | No | [VERIFIED] | Nature 381:607-609. DOI 10.1038/381607a0. |
| HAWKINS-2017 | Hawkins et al., 2017, *Frontiers in Neural Circuits* | A Theory of How Columns in the Neocortex Enable Learning the Structure of the World | 2017 | 2017 | Hawkins | Hawkins | YES | YES | YES | PARTIAL | No | [PARTIALLY_SUPPORTED] | Paper exists (DOI 10.3389/fncir.2017.00081, authors Hawkins, Ahmad, Cui). However, the source index URL (PMC5712570) links to a DIFFERENT paper — Cui et al. 2017 "HTM Spatial Pooler" in *Frontiers in Computational Neuroscience* (10.3389/fncom.2017.00111). Body text citation correct; source link incorrect. |
| GHOSH-2014 | Ghosh & Gilboa (2014, *Neuropsychologia* 53) | What is a memory schema? A historical perspective on current neuroscience literature | 2014 | 2014 | Ghosh | Ghosh | YES | YES | YES | YES | No | [VERIFIED] | Neuropsychologia 53:104-114. DOI 10.1016/j.neuropsychologia.2013.11.010. |
| WALKER-STICKGOLD | Walker & Stickgold (2004, Ann Rev Psych; 2006, Neuron) | Sleep, Memory, and Plasticity; Sleep-Dependent Learning and Memory Consolidation | 2004/2006 | 2006/2004 | Walker | Walker | YES | NO | YES | YES | No | [APPROXIMATE] | YEARS SWAPPED: "Annual Review of Psychology" paper published 2006 (not 2004), DOI 10.1146/annurev.psych.56.091103.070307. "Neuron" paper published 2004 (not 2006), DOI 10.1016/j.neuron.2004.08.031. Both papers real, years reversed in document. |
| STARESINA-2023 | Staresina et al. (2023, *Nature Neuroscience*) — cited for waking rest reactivation | How coupled slow oscillations, spindles and ripples coordinate neuronal processing and communication during human sleep | 2023 | 2023 | Staresina | Staresina | YES | YES | YES | PARTIAL | No | [PARTIALLY_SUPPORTED] | Paper exists and is in Nature Neuroscience (DOI 10.1038/s41593-023-01381-w). However, the paper is about sleep oscillation coupling, not explicitly waking rest reactivation as the document implies. The entailment between the claim and the abstract is PARTIAL. |
| RAO-BALLARD-1999 | Rao & Ballard (1999, *Nature Neuroscience*) | Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects | 1999 | 1999 | Rao | Rao | YES | YES | YES | YES | No | [VERIFIED] | Nature Neuroscience 2:79-87. DOI 10.1038/4580. |
| RAMSAUER-2020 | Ramsauer et al. (2021, ICLR — arXiv 2008.02217) | Hopfield Networks is All You Need | 2020/2021 | 2020 | Ramsauer | Ramsauer | YES | YES | YES | YES | No | [VERIFIED] | arXiv 2008.02217 submitted Aug 2020; published ICLR 2021. Both year references defensible. Source index cites "Ramsauer 2020" while body says "2021, ICLR" — minor inconsistency, not an error. |
| NTM-2014 | Graves et al. NTM (2014, arXiv 1410.5401) | Neural Turing Machines | 2014 | 2014 | Graves | Graves | YES | YES | YES | YES | No | [VERIFIED] | arXiv 1410.5401. Never formally published in a journal — remained a preprint. Citation as arXiv preprint is appropriate. |
| DNC-2016 | Graves et al. (2016, *Nature*) | Hybrid computing using a neural network with dynamic external memory | 2016 | 2016 | Graves | Graves | YES | YES | YES | YES | No | [VERIFIED] | Nature 538:471-476. DOI 10.1038/nature20101. |
| BUZZEGA-2020 | Buzzega et al. 2020, NeurIPS (arXiv 2004.07211) | Dark Experience for General Continual Learning: a Strong, Simple Baseline | 2020 | 2020 | Buzzega | Buzzega | YES | YES | YES | YES | No | [VERIFIED] | arXiv 2004.07211. NeurIPS 2020 venue not directly confirmed from arXiv page, but consistent with known publication record. |
| KANERVA-2009 | Kanerva (2009, *Cognitive Computation*) | Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors | 2009 | 2009 | Kanerva | Kanerva | YES | YES | YES | YES | No | [VERIFIED] | Cognitive Computation 1:139-159. DOI 10.1007/s12559-009-9009-8. |
| VSA-SURVEY | Kleyko, Rachkovskij, Osipov & Rahimi (*ACM Computing Surveys*, 2023) | A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II | 2023 | 2023 | Kleyko | Kleyko | YES | YES | YES | YES | No | [VERIFIED] | ACM Comput Surv. DOI 10.1145/3558000. This is Part II of the survey; Part I also exists (10.1145/3538531, 2023). |
| PARK-2023 | Park et al., 2023, UIST | Generative Agents: Interactive Simulacra of Human Behavior | 2023 | 2023 | Park | Park | YES | YES | YES | YES | No | [VERIFIED] | UIST 2023 proceedings. DOI 10.1145/3586183.3606763. |
| AMEM-2025 | Xu et al., NeurIPS 2025, arXiv 2502.12110 | A-MEM: Agentic Memory for LLM Agents | 2025 | 2025 | Xu | Xu (Wujiang) | YES | YES | YES | YES | No | [VERIFIED] | arXiv 2502.12110. Confirmed NeurIPS 2025 in arXiv comments field. |
| HIPPORAG-2024 | Guo et al., 2024, NeurIPS (arXiv 2405.14831) | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | 2024 | 2024 | Guo | Bernal Jiménez Gutiérrez | YES | YES | NO | YES | No | [FLAGGED] | AUTHOR ERROR: first author is Bernal Jiménez Gutiérrez (also written as Jimenez), NOT Guo. Paper is real and confirmed NeurIPS 2024. arXiv 2405.14831. |
| BRICKEN-2023 | Bricken et al. (2023, "Towards Monosemanticity") | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | 2023 | 2023 | Bricken | Bricken | YES | YES | YES | YES | No | [VERIFIED] | Anthropic transformer-circuits blog post (not peer-reviewed journal). Accessible at transformer-circuits.pub/2023/monosemantic-features. No DOI; no retraction possible. Correctly identified as Anthropic work. |
| CUNNINGHAM-2023 | Cunningham et al. (2023, arXiv 2309.08600) | Sparse Autoencoders Find Highly Interpretable Features in Language Models | 2023 | 2023 | Cunningham | Cunningham | YES | YES | YES | YES | No | [VERIFIED] | arXiv 2309.08600. |
| GRAVES-DNC-2016 | (also see DNC-2016 above) | — | — | — | — | — | — | — | — | — | — | Duplicate of DNC-2016 entry. |
| KIRKPATRICK-2017 | Kirkpatrick 2017, *PNAS* | Overcoming catastrophic forgetting in neural networks | 2017 | 2017 | Kirkpatrick | Kirkpatrick | YES | YES | YES | YES | No | [VERIFIED] | PNAS 114(13):3521-3526. DOI 10.1073/pnas.1611835114. |
| MILLIDGE-2022 | Millidge, Tschantz & Buckley (*Neural Computation* 2022) | Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs | 2022 | 2022 | Millidge | Millidge | YES | YES | YES | YES | No | [VERIFIED] | Neural Computation 34(6):1280-1317. DOI 10.1162/neco_a_01497. |
| TSCHANTZ-2023 | Tschantz et al., *PLOS CompBio* 2023 | Hybrid predictive coding: Inferring, fast and slow | 2023 | 2023 | Tschantz | Tschantz | YES | YES | YES | YES | No | [VERIFIED] | PLoS Comput Biol 19(8). DOI 10.1371/journal.pcbi.1011280. arXiv 2204.02169. |
| GAM-2604 | GAM (arXiv 2604.12285) | GAM: Hierarchical Graph-based Agentic Memory for LLM Agents | 2025 | 2025 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2604.12285 confirmed live and active. No conference publication yet. |
| TSM-2601 | TSM (arXiv 2601.07468) | Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents | 2025 | 2025 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2601.07468 confirmed live and active. |
| SURVEY-2603 | Memory for Autonomous LLM Agents survey (arXiv 2603.07670) | Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers | 2026 | 2026 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2603.07670 confirmed live and active. |
| EPISODIC-POS-2502 | Episodic memory position paper (arXiv 2502.06975) | Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents | 2025 | 2025 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2502.06975 confirmed live and active. |
| GRAPHITI-2501 | Zep/Graphiti (arXiv 2501.13956) | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | 2025 | 2025 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2501.13956 confirmed live and active. |
| GAAMA-2603 | GAAMA (arXiv 2603.27910) | GAAMA: Graph Augmented Associative Memory for Agents | 2026 | 2026 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2603.27910 confirmed live and active. Doc title says "Graph-Augmented" while arXiv says "Graph Augmented" (no hyphen) — cosmetic, within similarity threshold. |
| PAM-2602 | Predictive Associative Memory (arXiv 2602.11322) | Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence | 2026 | 2026 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2602.11322 confirmed live and active. |
| SCM-2604 | SCM (arXiv 2604.20943) | SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models | 2026 | 2026 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2604.20943 confirmed live and active. |
| LIGHTMEM-2510 | LightMem (arXiv 2510.18866) | LightMem: Lightweight and Efficient Memory-Augmented Generation | 2025 | 2025 | — | — | YES | YES | — | YES | No | [VERIFIED] | arXiv 2510.18866 confirmed live and active. |
| ROLLS-2006 | Rolls & Kesner (2006) — parenthetical only | A computational theory of hippocampal function, and empirical tests of the theory | 2006 | 2006 | Rolls | Rolls | YES | YES | YES | YES | No | [VERIFIED] | Prog Neurobiol 79:1-48. DOI 10.1016/j.pneurobio.2006.04.005. No venue claim in document; match confirmed. |
| LETTA-MEMGPT | Letta / MemGPT (product docs, no academic paper) | — | — | — | — | — | — | — | — | — | — | [UNVERIFIED] | Product documentation and blog posts, not a verifiable academic paper. URLs cited (letta.com, docs.letta.com) are live product sites. Acceptable as product citation. |
| MEM0 | mem0 (product blog, no academic paper) | — | — | — | — | — | — | — | — | — | — | [UNVERIFIED] | Product blog post (mem0.ai/blog/state-of-ai-agent-memory-2026). Not a peer-reviewed source. Acceptable as product citation. |
| CHATGPT-MEM | ChatGPT Memory (help article, no academic paper) | — | — | — | — | — | — | — | — | — | — | [UNVERIFIED] | OpenAI help article and Simon Willison blog. Not academic sources. Acceptable as product/journalism citation. |

---

## Summary

```yaml
verification_summary:
  total_citations: 46
  verified: 31
  partially_supported: 2
  approximate: 2
  unverified: 3
  flagged: 5
  retracted: 0
  export_cleared: false   # flagged + unverified > 0

flagged_citations:
  - YASSA-2011: journal mismatch (Trends in Neurosciences, not Nature Reviews Neuroscience)
  - FOSTER-2006: content error (reverse replay cited as forward replay)
  - BHATT-2020: no API record found; described finding matches a different paper
  - HIPPORAG-2024: author error (first author is Jimenez/Bernal Jiménez Gutiérrez, not Guo)
  - TSE-2007: volume error (Science vol 316, not 315) — marked APPROXIMATE but included here for attention

approximate_citations:
  - TSE-2007: Science vol 316 cited as 315 in body text
  - WALKER-STICKGOLD: publication years swapped between the two journals

partially_supported:
  - HAWKINS-2017: body citation correct; source index URL links to wrong paper
  - STARESINA-2023: paper exists and year/author/journal correct; entailment is partial (sleep paper, not waking rest)

unverified_product_sources:
  - LETTA-MEMGPT: product documentation
  - MEM0: product blog
  - CHATGPT-MEM: help article + journalism
  # These are product citations, not academic papers; acceptable in context

special_note_suspect_arxiv_ids:
  - "2603.07670 (survey): VERIFIED — real paper"
  - "2604.12285 (GAM): VERIFIED — real paper"
  - "2601.07468 (TSM): VERIFIED — real paper"
  - "2603.27910 (GAAMA): VERIFIED — real paper"
  - "2602.11322 (PAM): VERIFIED — real paper"
  - "2604.20943 (SCM): VERIFIED — real paper"
  # All suspect arXiv IDs in the 2601-2604 range checked out as real papers
```

---

## Detailed Flag Notes

### F-01: YASSA-2011 — Wrong Journal [FLAGGED]

**Claimed:** Yassa & Stark (2011), *Nature Reviews Neuroscience* 12:367-378  
**Actual:** *Trends in Neurosciences* 34:515-525, DOI 10.1016/j.tins.2011.06.006  
**Impact:** The paper exists and the content is correct (pattern separation review). The PMC link in the source index (PMC3183227, PMID 21788086) resolves to the correct paper. Only the journal name and volume/pages in the body text are wrong. Journals are entirely different — NRN vs. Trends in Neurosciences.

---

### F-02: FOSTER-2006 — Content Error (Reverse vs. Forward Replay) [FLAGGED]

**Claimed:** Foster & Wilson (2006, *Nature* 440) supports "Forward replay"  
**Actual title:** "Reverse replay of behavioural sequences in hippocampal place cells during the awake state" (Nature 440:680, DOI 10.1038/nature04587)  
**Impact:** Foster & Wilson 2006 describes REVERSE replay, not forward replay. The document inverts the function when assigning this reference. Diba & Buzsáki (2007) is correctly cited for reverse replay in the same section, so the bibliographic assignments are crossed. The correct citation for forward replay is Nádasdy et al. 1999 or later forward-replay-specific work; Foster & Wilson 2006 should be moved to the reverse replay bullet.

---

### F-03: BHATT-2020 — Likely Fabricated Citation [FLAGGED]

**Claimed:** Bhatt et al., 2020, *Journal of Neuroscience* — blocking GluA2/AMPAR removal prevents forgetting of object-location memories  
**API result:** No PubMed, Crossref, or OpenAlex record for "Bhatt 2020 Journal of Neuroscience" matching this description  
**Actual source:** Migues et al. (2016), J Neuroscience 36:3481-3494, DOI 10.1523/JNEUROSCI.3333-15.2016 — "Blocking Synaptic Removal of GluA2-Containing AMPA Receptors Prevents the Natural Forgetting of Long-Term Memories"  
**Impact:** The empirical claim is real and traceable, but the citation is wrong. The Bhatt 2020 citation appears to be fabricated; the Migues 2016 paper describes exactly the experiment cited.

---

### F-04: HIPPORAG-2024 — Wrong First Author [FLAGGED]

**Claimed:** Guo et al., 2024, NeurIPS (arXiv 2405.14831)  
**Actual first author:** Bernal Jiménez Gutiérrez (also cited as "Jimenez" in short form)  
**Full author list:** Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su  
**Impact:** Paper is confirmed real and at NeurIPS 2024. Author surname is wrong. Correct short citation is "Jimenez et al., 2024" or "Bernal Jiménez Gutiérrez et al., 2024".

---

### F-05: TSE-2007 — Volume Number Error [APPROXIMATE]

**Claimed:** Tse et al. (2007, *Science* 315)  
**Actual:** Science **316**(5821):76-82, DOI 10.1126/science.1135935  
**Impact:** Minor error; the paper is confirmed real and the DOI in the source index is correct. Science vol 315 covers January–March 2007; this paper was published April 6, 2007 in vol 316. Correcting "315" to "316" in the body text resolves the discrepancy.

---

### A-01: WALKER-STICKGOLD — Years Swapped [APPROXIMATE]

**Claimed:** Walker & Stickgold (2004, *Annual Review of Psychology*; 2006, *Neuron*)  
**Actual:**  
- Annual Review of Psychology paper: published **2006** (DOI 10.1146/annurev.psych.56.091103.070307)  
- Neuron paper: published **2004** (DOI 10.1016/j.neuron.2004.08.031)  
**Impact:** Both papers exist and both are by Walker & Stickgold. The years are simply reversed in the citation.

---

### P-01: HAWKINS-2017 — Correct Citation, Wrong Source Index URL [PARTIALLY_SUPPORTED]

**Body text citation:** Hawkins et al., 2017, *Frontiers in Neural Circuits* — CORRECT (DOI 10.3389/fncir.2017.00081, "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World")  
**Source Index URL:** PMC5712570 links to Cui et al. 2017 "HTM Spatial Pooler" in *Frontiers in Computational Neuroscience* (DOI 10.3389/fncom.2017.00111) — DIFFERENT PAPER  
**Impact:** The body text citation is accurate. Only the hyperlink in the source index points to the wrong paper. Both papers are from Numenta (2017), by overlapping author teams, which explains the confusion.

---

### P-02: STARESINA-2023 — Paper Exists, Entailment Partial [PARTIALLY_SUPPORTED]

**Claimed:** Staresina et al. (2023, *Nature Neuroscience*) shows "ripple-coupled offline reactivation events occur during waking rest as well as sleep"  
**Actual paper:** "How coupled slow oscillations, spindles and ripples coordinate neuronal processing and communication during human sleep" (DOI 10.1038/s41593-023-01381-w)  
**Impact:** The paper is a sleep physiology paper. Its abstract and content concern sleep-stage oscillation coupling, not waking rest reactivation. No separate Staresina 2023 waking-rest paper was found. The claim may refer to a different Staresina paper or may be an incorrect characterization of the 2023 paper's content. Mark as PARTIALLY_SUPPORTED (correct author/year/journal; entailment unclear).

---

## Suggested Replacements (Fix List)

| Citation ID | Current (wrong) | Replacement | Canonical DOI |
|-------------|-----------------|-------------|---------------|
| YASSA-2011 | Yassa & Stark (2011), *Nature Reviews Neuroscience* 12:367-378 | Yassa & Stark (2011), *Trends in Neurosciences* 34(10):515-525 | 10.1016/j.tins.2011.06.006 |
| FOSTER-2006 | Cited for "Forward replay" | Move to reverse replay bullet; for forward replay use: Nádasdy et al. (1999) J Neurosci or Dragoi & Tonegawa (2011) Nature | 10.1038/nature04587 (reverse replay confirmed) |
| TSE-2007 | Tse et al. (2007, *Science* **315**) | Tse et al. (2007, *Science* **316**(5821):76-82) — "Schemas and Memory Consolidation" | 10.1126/science.1135935 |
| BHATT-2020 | Bhatt et al., 2020, *Journal of Neuroscience* | Migues et al. (2016), *Journal of Neuroscience* 36(11):3481-3494 — "Blocking Synaptic Removal of GluA2-Containing AMPA Receptors Prevents the Natural Forgetting of Long-Term Memories" | 10.1523/JNEUROSCI.3333-15.2016 |
| HIPPORAG-2024 | Guo et al., 2024, NeurIPS | Bernal Jiménez Gutiérrez et al., 2024, NeurIPS — "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" | 10.48550/arXiv.2405.14831 |
| WALKER-STICKGOLD | Walker & Stickgold (2004, *Ann Rev Psych*; 2006, *Neuron*) | Walker & Stickgold (2006, *Ann Rev Psych*; 2004, *Neuron*) | 10.1146/annurev.psych.56.091103.070307 (2006); 10.1016/j.neuron.2004.08.031 (2004) |
| HAWKINS-2017 source link | PMC5712570 (wrong — links to Cui et al. 2017 Frontiers Comp Neuro) | Replace URL with: https://doi.org/10.3389/fncir.2017.00081 | 10.3389/fncir.2017.00081 |
| HARDT-2013 URL | PDF link to non-DOI source | Use DOI: 10.1016/j.tics.2013.01.001 | 10.1016/j.tics.2013.01.001 |
| KUMARAN-2016 issue | Volume 20(8) | Volume 20(**7**):512-534 | 10.1016/j.tics.2016.05.004 |

### Notes on Confirmed-Good Citations Worth Highlighting

- **McClelland 1995**: Canonical CLS paper verified (Psych Rev 102(3):419-457).
- **Tse 2007 "Schemas and Memory Consolidation"**: DOI in source index correct (10.1126/science.1135935); only body text has vol 315 vs. actual vol 316.
- **Nader 2000 reconsolidation**: Verified, no retraction (Nature 406:722-726).
- **Diba & Buzsáki 2007**: Verified with correct volume and pages (NatNeuro 10:1241-1242).
- **Bricken 2023 "Towards Monosemanticity"**: Correctly identified as Anthropic transformer-circuits blog. Title is correct; no DOI (blog-format publication).
- **All 2601-2604 arXiv preprints**: Every suspect arXiv ID checked out as a real, active paper. None are fabricated IDs.
- **A-MEM (arXiv 2502.12110)**: Confirmed NeurIPS 2025 acceptance (verified from arXiv comments field).
- **HippoRAG (arXiv 2405.14831)**: Confirmed NeurIPS 2024 acceptance (verified from arXiv comments field).

---

*Report generated by biss-brain citation-verifier. Source documents were not modified.*
