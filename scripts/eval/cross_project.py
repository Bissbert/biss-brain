"""Cross-project eval: what the shared pool gives other projects, and what it must not.

  run_eval.py --suite synthetic --cross-project [--embed fake|ollama] [--out FILE]

Builds the synthetic brain plus tests/fixtures/eval_crossproj (general facts,
facts tagged general that carry an identifier, project facts, and project
facts wrongly tagged general), runs the dream promotion into `_global`, then
asks every question from its own project through the shipped read paths:
retrieve.retrieve / prompt_inject (local), retrieve.global_candidates at the
recall gate (recall CLI and MCP) and retrieve.pool_inject (prompt hook).

Reports four groups, ids and numbers only:
  leak        planted identifier facts that got into the pool (veto escapes),
              project facts in the pool, must-not-transfer answers returned by
              the recall fan-out or injected, poison injected
  benefit     cross_project questions answered by the pool (hit@1/3, MRR),
              by the local store alone, and by the pool inject
  regression  local metrics are unchanged by construction (pool rows are a
              separate list and only fill free inject slots); measured instead:
              pool rows shown or injected on questions with a local answer and
              pool injections on no-answer / off-topic prompts
  latency     global_candidates per prompt, opening the pool each time as the
              hook does; Δp95 of retrieve + pool against retrieve alone

The scope tags in the corpus are set by hand: this measures the veto and the
gates, not the distiller's tagging.
"""
from __future__ import annotations

import copy
import sqlite3
import time
from pathlib import Path

CROSS_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_crossproj"
LEAK_TYPES = ("no_transfer_veto", "no_transfer_local", "no_transfer_mistag")
GATE_SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9)


def corpus_roles() -> dict[str, str]:
    import json  # noqa: PLC0415

    return {c["id"]: c.get("role", "") for line in (CROSS_DIR / "corpus.jsonl").read_text().splitlines()
            if line.strip() for c in [json.loads(line)]}


def promote(root: Path, cfg: dict) -> dict:
    """Run the dream promotion for every store; per project: promoted count and veto reasons."""
    import dream  # noqa: PLC0415

    out: dict[str, dict] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_global"):
        n = dream._stage_promote_to_global(d, cfg)
        c = sqlite3.connect(str(d / "meta.sqlite"))
        try:
            vetoes = dict(c.execute("SELECT global_veto, COUNT(*) FROM chunks WHERE global_veto != '' "
                                    "GROUP BY global_veto").fetchall())
        finally:
            c.close()
        out[d.name] = {"promoted": n, "vetoed": vetoes}
    return out


def pool_sources(root: Path) -> set[str]:
    """Source ids of the live pool rows."""
    path = root / "_global" / "meta.sqlite"
    if not path.exists():
        return set()
    c = sqlite3.connect(str(path))
    try:
        return {r[0] for r in c.execute("SELECT source_id FROM chunks WHERE superseded_by = ''")}
    finally:
        c.close()


def run(root: Path, questions: list[dict], cfg: dict, emb, k: int) -> dict:
    import retrieve  # noqa: PLC0415
    from eval_metrics import is_answerable, relevance_vector  # noqa: PLC0415
    from run_eval import open_meta_ro, open_table  # noqa: PLC0415

    xp = cfg["cross_project"]
    gate = float(xp.get("recall", {}).get("cosine_gate", 0.85))
    icfg = copy.deepcopy(cfg)
    icfg["cross_project"]["inject"] = True
    poison = {pid for q in questions for pid in q.get("poison_ids", [])}
    tables: dict[str, object] = {}
    metas: dict[str, object] = {}
    per_q = {}
    for q in questions:
        proj = q["project"]
        if proj not in tables:
            tables[proj] = open_table(root, proj)
            metas[proj] = open_meta_ro(root, proj) if tables[proj] is not None else None
        tbl, meta = tables[proj], metas[proj]
        qvec = emb.embed([q["query"]], "query")[0]
        t0 = time.perf_counter()
        local = retrieve.retrieve(tbl, meta, proj, q["query"], qvec, cfg, top_k=k) if tbl is not None else []
        injected = retrieve.prompt_inject(tbl, meta, proj, q["query"], qvec, cfg) if tbl is not None else []
        t1 = time.perf_counter()
        pool = retrieve.global_candidates(q["query"], qvec, cfg, k, min_cosine=gate)
        t2 = time.perf_counter()
        pinj = (retrieve.pool_inject(q["query"], qvec, icfg, proj, injected)
                if not retrieve.trivial_prompt(q["query"], cfg) else [])
        # cosine of the best answering pool row, whatever the gate (for the sweep)
        everything = retrieve.global_candidates(q["query"], qvec, cfg, 50, min_cosine=-1.0)
        ans_cos = [r["_cosine"] for r, rel in zip(everything, relevance_vector(everything, q)) if rel]
        per_q[q["id"]] = {
            "type": q.get("type", ""),
            "answerable": is_answerable(q),
            "local_rel": relevance_vector(local, q),
            "inject_ids": [r["id"][:12] for r in injected],
            "inject_rel": relevance_vector(injected, q),
            "pool_ids": [r["id"][:12] for r in pool],
            "pool_rel": relevance_vector(pool, q),
            "pool_inject_ids": [r["id"][:12] for r in pinj],
            "pool_inject_rel": relevance_vector(pinj, q),
            "answer_cos": round(max(ans_cos), 4) if ans_cos else None,
            "top_pool_cos": round(everything[0]["_cosine"], 4) if everything else None,
            "poison_injected": sum(1 for r in injected + pinj if r["id"] in poison),
            "retrieve_ms": (t1 - t0) * 1000,
            "global_ms": (t2 - t1) * 1000,
        }
    for m in metas.values():
        if m is not None:
            m.close()
    return per_q


def summarise(per_q: dict, roles: dict[str, str], in_pool: set[str], promoted: dict, cfg: dict, k: int) -> dict:
    from eval_metrics import hit_at_k, mean, reciprocal_rank  # noqa: PLC0415

    qs = list(per_q.values())
    cross = [p for p in qs if p["type"] == "cross_project"]
    local_ans = [p for p in qs if p["answerable"] and p["type"] not in ("cross_project", *LEAK_TYPES)]
    quiet = [p for p in qs if not p["answerable"] or p["type"] == "off_topic"]
    gate = float(cfg["cross_project"].get("recall", {}).get("cosine_gate", 0.85))
    igate = float(cfg["cross_project"].get("inject_min_cosine", 0.88))

    def pct(xs, f):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 2) if xs else None

    def leaked(t: str, key: str) -> int:
        return sum(1 for p in qs if p["type"] == t and any(p[key]))

    cos = [p["answer_cos"] for p in cross if p["answer_cos"] is not None]
    by_role = {r: sorted(i for i, role in roles.items() if role == r) for r in set(roles.values())}
    return {
        "promotion": {
            "promoted": sum(v["promoted"] for v in promoted.values()),
            "vetoed": {r: sum(v["vetoed"].get(r, 0) for v in promoted.values())
                       for r in sorted({r for v in promoted.values() for r in v["vetoed"]})},
            "pool_rows": len(in_pool),
            "general_pooled": f"{sum(1 for i in by_role.get('general', []) if i in in_pool)}"
                              f"/{len(by_role.get('general', []))}",
        },
        "leak": {
            "veto_escapes": sorted(i for i in by_role.get("veto", []) if i in in_pool),
            "project_facts_in_pool": sorted(i for i in by_role.get("local", []) if i in in_pool),
            "recall_leak_veto": leaked("no_transfer_veto", "pool_rel"),
            "recall_leak_local": leaked("no_transfer_local", "pool_rel"),
            "inject_leak_veto": leaked("no_transfer_veto", "pool_inject_rel"),
            "inject_leak_local": leaked("no_transfer_local", "pool_inject_rel"),
            # tagged general by mistake, no identifier: the veto cannot see these
            "mistag_pooled": sorted(i for i in by_role.get("mistag", []) if i in in_pool),
            "mistag_recalled": leaked("no_transfer_mistag", "pool_rel"),
            "mistag_injected": leaked("no_transfer_mistag", "pool_inject_rel"),
            "poison_injected": sum(p["poison_injected"] for p in qs),
        },
        "benefit": {
            "n": len(cross),
            "local_hit@k": mean(hit_at_k(p["local_rel"], k) for p in cross),
            "pool_hit@1": mean(hit_at_k(p["pool_rel"], 1) for p in cross),
            "pool_hit@3": mean(hit_at_k(p["pool_rel"], 3) for p in cross),
            "pool_mrr": mean(reciprocal_rank(p["pool_rel"]) for p in cross),
            "pool_inject_hit": mean(float(any(p["pool_inject_rel"])) for p in cross),
            "answer_cos_min_median_max": (pct(cos, 0.0), pct(cos, 0.5), pct(cos, 1.0)) if cos else None,
            # share of cross_project questions whose answer clears each gate
            "gate_sweep": {g: round(sum(1 for c in cos if c >= g) / len(cross), 3) for g in GATE_SWEEP}
            if cross else {},
            "recall_gate": gate, "inject_gate": igate,
        },
        "regression": {
            "local_mrr_change": "0 by construction (pool rows are listed separately)",
            "n_local_answerable": len(local_ans),
            "pool_shown_on_local_questions": mean(float(bool(p["pool_ids"])) for p in local_ans),
            "pool_injected_on_local_questions": mean(float(bool(p["pool_inject_ids"])) for p in local_ans),
            "pool_false_inject_rate": mean(float(bool(p["pool_inject_ids"])) for p in quiet) if quiet else None,
            "n_quiet": len(quiet),
        },
        "latency": {
            "global_p50_ms": pct([p["global_ms"] for p in qs], 0.5),
            "global_p95_ms": pct([p["global_ms"] for p in qs], 0.95),
            "retrieve_p95_ms": pct([p["retrieve_ms"] for p in qs], 0.95),
            "with_pool_p95_ms": pct([p["retrieve_ms"] + p["global_ms"] for p in qs], 0.95),
        },
    }


def main_run(root: Path, emb, cfg0: dict, k: int, offtopic: bool) -> dict:
    """Build, promote, ask; the summary plus the per-question records."""
    import run_eval  # noqa: PLC0415

    names = run_eval.build_synthetic(root, emb, extra=(CROSS_DIR / "corpus.jsonl",))
    questions = (run_eval.load_questions(run_eval.SYNTHETIC_DIR / "questions.jsonl", names)
                 + run_eval.load_questions(CROSS_DIR / "questions.jsonl", names))
    if offtopic:
        questions = run_eval.with_offtopic(questions)
    cfg = copy.deepcopy(cfg0)
    cfg["cross_project"]["enabled"] = True
    cfg["cross_project"]["inject"] = False
    cfg["embed"]["dim"] = len(emb.embed(["dimension probe"], "doc")[0])
    promoted = promote(root, cfg)
    per_q = run(root, questions, cfg, emb, k)
    summary = summarise(per_q, corpus_roles(), pool_sources(root), promoted, cfg, k)
    return {"summary": summary, "promoted": promoted, "per_question": per_q}


def print_summary(s: dict) -> None:
    for group in ("promotion", "leak", "benefit", "regression", "latency"):
        print(f"\n[{group}]")
        for key, v in s[group].items():
            print(f"  {key:34} {v}")
