#!/usr/bin/env python3
"""biss-brain recall eval harness.

  run_eval.py --suite synthetic [--embed fake|ollama] [--variants ...]
  run_eval.py --suite QUESTIONS.jsonl --brain SNAPSHOT_DIR [--embed-models m1,m2+prefix]

Runs every question through the production ranking code in scripts/retrieve.py
(the same functions the recall CLI, MCP server and UserPromptSubmit hook call)
against a brain snapshot, never the live brain. Reports hit@k, MRR, nDCG@k
with bootstrap CIs, injection precision / false-injection rate, latency, and
paired-bootstrap deltas against a baseline variant.

Output contains question ids, chunk ids and numbers only, never memory text.
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LANCE_LOG", "error")
os.environ.setdefault("RUST_LOG", "error")

EVAL_DIR = Path(__file__).resolve().parent
SCRIPTS = EVAL_DIR.parent
SYNTHETIC_DIR = SCRIPTS.parent / "tests" / "fixtures" / "eval_synthetic"
SYNTHETIC_BASE_TS = 1_750_000_000  # corpus "day" N is this + N days


class RootGuardError(RuntimeError):
    pass


def live_roots() -> set[Path]:
    """The live brain, from $HOME and from the passwd entry (HOME can be faked)."""
    homes = {Path.home(), Path(pwd.getpwuid(os.getuid()).pw_dir)}
    return {(h / ".claude" / "brain").resolve() for h in homes}


def guard_root(root: Path) -> Path:
    """Refuse the live brain (or anything inside it) as an eval root."""
    r = root.expanduser().resolve()
    if any(r == live or live in r.parents for live in live_roots()):
        raise RootGuardError(f"refusing to evaluate against the live brain at {r}; snapshot it first")
    return r


def _bind_root(root: Path) -> None:
    """Point the brain modules at *root*. Must run before importing common."""
    if "common" in sys.modules:
        raise RootGuardError("common was imported before the eval root was bound")
    os.environ["BISS_BRAIN_ROOT"] = str(root)
    cfg = root / "config.toml"
    if cfg.exists():
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg)
    else:
        os.environ.pop("BISS_BRAIN_CONFIG", None)
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(EVAL_DIR))
    import common  # noqa: PLC0415

    if Path(common.BRAIN_ROOT).resolve() != root:
        raise RootGuardError(f"common.BRAIN_ROOT={common.BRAIN_ROOT} != eval root {root}")


# ── variants ─────────────────────────────────────────────────────────────
# A variant is a config overlay on the snapshot's own config plus the table
# set it reads ("snapshot" = tables as stored; any other = rebuilt tables).

BASE_VARIANTS: dict[str, dict] = {
    "prod": {},
    "dense": {"retrieve": {"hybrid": False}},
    "no_decay": {"retrieve": {"decay_weight": 0.0}},
    "dense_no_decay": {"retrieve": {"hybrid": False, "decay_weight": 0.0}},
    **{f"dw{w}": {"retrieve": {"decay_weight": w}} for w in (0.05, 0.1, 0.2, 0.5, 1.0)},
    # candidate-pool ablation (v0.6 fetched exactly top_k into RRF)
    **{f"pool{n}": {"retrieve": {"pool_factor": 1, "pool_min": n, "decay_weight": 0.0}} for n in (5, 10, 15, 25, 50)},
    "dense_pool5": {"retrieve": {"hybrid": False, "pool_factor": 1, "pool_min": 5, "decay_weight": 0.0}},
    # injection gate sweep: tune inject_min_cosine against false injections
    **{f"inj{c}": {"retrieve": {"inject_min_cosine": c}} for c in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)},
    # prompt injection when the embed missed its budget (BM25-only candidates)
    "novec": {"_eval": {"no_vector": True}},
    "summary_first": {"dream": {"prefer_summaries_in_recall": True}},
    "rerank": {"retrieve": {"rerank": True}},
    # v0.7 phase 7: provenance rank multiplier off (tool/web rows ranked like the rest)
    "no_trust": {"provenance": {"rank_multiplier": {"tool": 1.0, "web": 1.0}}},
}
DEFAULT_VARIANTS = ["prod", "dense", "no_decay", "dense_no_decay"]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ── embedders ────────────────────────────────────────────────────────────

class Embedder:
    """Ollama /api/embed client with optional nomic-style task prefixes and a cache."""

    PREFIXES = {
        "nomic-embed-text": ("search_query: ", "search_document: "),
        "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
        "snowflake-arctic-embed2": ("query: ", ""),
        "qwen3-embedding": ("Instruct: Given a question, retrieve memory notes that answer it\nQuery: ", ""),
        "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    }

    def __init__(self, spec: str, endpoint: str, fake_dim: int = 0):
        self.label = spec
        self.prefix = spec.endswith("+prefix")
        self.model = spec.removesuffix("+prefix")
        self.endpoint = endpoint
        self.fake_dim = fake_dim
        self.cache: dict[tuple[str, str], list[float]] = {}

    def _prefix(self, role: str) -> str:
        if not self.prefix:
            return ""
        q, d = self.PREFIXES.get(self.model.split(":")[0], ("", ""))
        return q if role == "query" else d

    def embed(self, texts: list[str], role: str) -> list[list[float]]:
        if self.fake_dim:
            from common import _fake_embed  # noqa: PLC0415

            return [_fake_embed(t, self.fake_dim) for t in texts]
        import httpx  # noqa: PLC0415

        todo = [t for t in texts if (role, t) not in self.cache]
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            r = httpx.post(
                f"{self.endpoint}/api/embed",
                json={"model": self.model, "input": [self._prefix(role) + t for t in batch], "truncate": True},
                timeout=600,
            )
            r.raise_for_status()
            for t, v in zip(batch, r.json()["embeddings"]):
                self.cache[(role, t)] = v
        return [self.cache[(role, t)] for t in texts]


# ── brain access ─────────────────────────────────────────────────────────

def open_table(root: Path, proj: str):
    import lancedb  # noqa: PLC0415

    ld = root / proj / "lancedb"
    if not ld.is_dir():
        return None
    db = lancedb.connect(str(ld))
    return db.open_table("chunks") if "chunks" in db.list_tables().tables else None


def open_meta_ro(root: Path, proj: str):
    import sqlite3  # noqa: PLC0415

    return sqlite3.connect(f"file:{root / proj / 'meta.sqlite'}?mode=ro", uri=True)


_ROW_FIELDS = ("id", "project", "session_id", "agent_id", "kind", "ts", "text", "content_hash", "superseded_by")


def rebuild_tables(src_root: Path, dest_root: Path, projects: list[str], emb: Embedder) -> None:
    """Copy each project's rows into a fresh table embedded with *emb*.

    The FTS index is built after the data, so every rebuilt table set has the
    same (complete) BM25 coverage and the embedding is the only variable.
    """
    import lancedb  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    for proj in projects:
        tbl = open_table(src_root, proj)
        if tbl is None:
            continue
        rows = tbl.search().select(list(_ROW_FIELDS)).limit(1_000_000).to_list()
        if not rows:
            continue
        vecs = emb.embed([r["text"] for r in rows], "doc")
        dim = len(vecs[0])
        out = dest_root / proj
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / proj / "meta.sqlite", out / "meta.sqlite")
        _migrate_meta(out / "meta.sqlite")
        schema = pa.schema(
            [pa.field(f, pa.int64() if f == "ts" else pa.string()) for f in _ROW_FIELDS]
            + [pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        db = lancedb.connect(str(out / "lancedb"))
        t = db.create_table("chunks", schema=schema, mode="overwrite")
        t.add([{**{f: r[f] for f in _ROW_FIELDS}, "vector": v} for r, v in zip(rows, vecs)])
        t.create_fts_index("text", replace=True)


def _migrate_meta(path: Path) -> None:
    """Apply the production schema to a copied meta db (adds v0.7 provenance defaults)."""
    import sqlite3  # noqa: PLC0415

    import common  # noqa: PLC0415

    conn = sqlite3.connect(path)
    try:
        common.ensure_schema(conn)
    finally:
        conn.close()


def build_synthetic(root: Path, emb: Embedder) -> dict[str, str]:
    """Materialise the committed synthetic corpus as a brain under *root*.

    Returns {project name: project hash}.
    """
    import hashlib  # noqa: PLC0415
    import sqlite3  # noqa: PLC0415

    import lancedb  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    import provenance  # noqa: PLC0415

    corpus = [json.loads(line) for line in (SYNTHETIC_DIR / "corpus.jsonl").read_text().splitlines() if line.strip()]
    names = sorted({c["project"] for c in corpus})
    hashes = {n: hashlib.sha256(f"synthetic/{n}".encode()).hexdigest()[:12] for n in names}
    base_ts = SYNTHETIC_BASE_TS
    # `supersedes` stands in for a CONTRADICTS verdict from the conflict judge.
    successor = {c["supersedes"]: c for c in corpus if c.get("supersedes")}
    for name in names:
        items = [c for c in corpus if c["project"] == name]
        proj = hashes[name]
        rows = []
        for c in items:
            rows.append({
                "id": c["id"], "project": proj, "session_id": c.get("session", ""), "agent_id": "",
                "kind": c.get("kind", "stop"), "ts": base_ts + int(c.get("day", 0)) * 86400,
                "text": c["text"], "content_hash": hashlib.sha256(c["text"].encode()).hexdigest(),
                "superseded_by": c.get("superseded_by") or (successor[c["id"]]["id"] if c["id"] in successor else ""),
            })
        vecs = emb.embed([r["text"] for r in rows], "doc")
        dim = len(vecs[0])
        (root / proj).mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [pa.field(f, pa.int64() if f == "ts" else pa.string()) for f in _ROW_FIELDS]
            + [pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        t = lancedb.connect(str(root / proj / "lancedb")).create_table("chunks", schema=schema, mode="overwrite")
        t.add([{**r, "vector": v} for r, v in zip(rows, vecs)])
        t.create_fts_index("text", replace=True)
        meta_path = root / proj / "meta.sqlite"
        _migrate_meta(meta_path)
        m = sqlite3.connect(meta_path)
        for c, r in zip(items, rows):
            # Corpus rows default to distilled-from-conversation (assistant).
            src = c.get("source", "assistant")
            prov = (provenance.derive(c["taint"], source=src) if "taint" in c
                    else provenance.make(src) if src in provenance.RAW_SOURCES
                    else provenance.derive(("assistant",), source=src))
            nxt = successor.get(c["id"])
            invalid_at = base_ts + int(nxt.get("day", 0)) * 86400 if nxt else None
            m.execute(
                "INSERT INTO chunks (id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, "
                "last_accessed, decay_score, pinned, source, trust, taint, invalid_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["id"], proj, r["session_id"], "", r["kind"], r["ts"], r["content_hash"], r["superseded_by"],
                 r["ts"], float(c.get("decay", 1.0)), int(c.get("pinned", 0)), *prov.row(), invalid_at),
            )
        m.commit()
        m.close()
    return hashes


# ── run ──────────────────────────────────────────────────────────────────

def load_questions(path: Path, name_to_hash: dict[str, str] | None = None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith("//")]
    for q in qs:
        if name_to_hash:
            q["project"] = name_to_hash.get(q["project"], q["project"])
    return qs


OFFTOPIC = SCRIPTS.parent / "tests" / "fixtures" / "eval_offtopic.jsonl"


def with_offtopic(questions: list[dict]) -> list[dict]:
    """Append the off-topic prompts once per project in the suite.

    They measure abstention (no injection for acks, commands and general
    questions); they never count as answerable or as no-answer questions.
    """
    extra = load_questions(OFFTOPIC)
    projects = sorted({q["project"] for q in questions})
    return questions + [{**o, "id": f"{o['id']}@{p[:8]}", "project": p} for p in projects for o in extra]


def corpus_coverage(root: Path, questions: list[dict]) -> dict:
    """Which answerable questions have at least one live (non-superseded) answer row
    (any row for an as_of question).

    Reported per run so lost answers (e.g. after re-distillation) show apart from
    lost ranking.
    """
    from eval_metrics import is_answerable, is_relevant  # noqa: PLC0415

    rows: dict[str, list[dict]] = {}
    missing = []
    for q in questions:
        if not is_answerable(q):
            continue
        proj = q["project"]
        if proj not in rows:
            tbl = open_table(root, proj)
            rows[proj] = [] if tbl is None else (
                tbl.search().select(["id", "text", "superseded_by"]).limit(1_000_000).to_list()
            )
        pool = rows[proj] if "as_of_day" in q else [r for r in rows[proj] if not r["superseded_by"]]
        if not any(is_relevant(r["text"], r["id"], q) for r in pool):
            missing.append(q["id"])
    n = sum(1 for q in questions if is_answerable(q))
    return {"answerable": n, "in_corpus": n - len(missing), "missing": missing}


def run_variant(
    root: Path, questions: list[dict], cfg: dict, emb: Embedder, k: int, inject: bool
) -> dict:
    import common  # noqa: PLC0415
    import retrieve  # noqa: PLC0415
    from eval_metrics import is_answerable, relevance_vector  # noqa: PLC0415

    per_q = {}
    # Poisoned memories (planted untrusted content) named by any question.
    poison = {pid for q in questions for pid in q.get("poison_ids", [])}
    tables: dict[str, object] = {}
    metas: dict[str, object] = {}
    fallbacks_before = common.HYBRID_FALLBACKS
    for q in questions:
        proj = q["project"]
        if proj not in tables:
            tables[proj] = open_table(root, proj)
            metas[proj] = open_meta_ro(root, proj) if tables[proj] is not None else None
        tbl, meta = tables[proj], metas[proj]
        t0 = time.perf_counter()
        qvec = emb.embed([q["query"]], "query")[0]
        t1 = time.perf_counter()
        rows: list[dict] = []
        injected: list[dict] = []
        cands: list[dict] = []
        error = None
        if tbl is not None:
            try:
                as_of = SYNTHETIC_BASE_TS + int(q["as_of_day"]) * 86400 if "as_of_day" in q else None
                rows = retrieve.retrieve(
                    tbl, meta, proj, q["query"], qvec, cfg, top_k=k,
                    summary_first=cfg.get("dream", {}).get("prefer_summaries_in_recall", False),
                    as_of=as_of,
                )
                if inject:
                    ivec = None if cfg.get("_eval", {}).get("no_vector") else qvec
                    injected = retrieve.prompt_inject(tbl, meta, proj, q["query"], ivec, cfg)
                    # pre-gate candidates and their gate features, for calibrate_inject.py
                    if not retrieve.trivial_prompt(q["query"], cfg):
                        cands = retrieve.inject_candidates(tbl, meta, proj, q["query"], ivec, cfg)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {str(e)[:120]}"
        t2 = time.perf_counter()
        per_q[q["id"]] = {
            "type": q.get("type", ""),
            "answerable": is_answerable(q),
            "ids": [r["id"][:12] for r in rows],
            "rel": relevance_vector(rows, q),
            "stale_top1": bool(rows) and any(
                all(t.lower() in rows[0]["text"].lower() for t in g) for g in q.get("stale", [])
            ),
            "inject_ids": [r["id"][:12] for r in injected],
            "inject_rel": relevance_vector(injected, q),
            "poison_injected": sum(1 for r in injected if r["id"] in poison),
            "inject_cands": [
                {"cos": round(float(r.get("_cosine", 0.0)), 4), "bm25": round(float(r.get("_bm25", 0.0)), 3),
                 "rel": rel, "poison": r["id"] in poison, "chars": len(r.get("text", ""))}
                for r, rel in zip(cands, relevance_vector(cands, q))
            ],
            "poison_top1": bool(rows) and rows[0]["id"] in poison,
            "embed_ms": (t1 - t0) * 1000,
            "retrieve_ms": (t2 - t1) * 1000,
            "error": error,
        }
    for m in metas.values():
        if m is not None:
            m.close()
    return {"per_question": per_q, "hybrid_fallbacks": common.HYBRID_FALLBACKS - fallbacks_before}


def summarise(res: dict, k: int) -> dict:
    from eval_metrics import bootstrap_ci, hit_at_k, mean, ndcg_at_k, reciprocal_rank  # noqa: PLC0415

    pq = res["per_question"].values()
    ans = [p for p in pq if p["answerable"]]
    off = [p for p in pq if p.get("type") == "off_topic"]
    noans = [p for p in pq if not p["answerable"] and p.get("type") != "off_topic"]
    mrr = [reciprocal_rank(p["rel"]) for p in ans]
    inj = [x for p in ans for x in p["inject_rel"]]
    lat = sorted(p["embed_ms"] + p["retrieve_ms"] for p in pq)
    ret = sorted(p["retrieve_ms"] for p in pq)

    def pct(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")

    return {
        "n_answerable": len(ans),
        "n_no_answer": len(noans),
        "hit@1": mean(hit_at_k(p["rel"], 1) for p in ans),
        "hit@3": mean(hit_at_k(p["rel"], 3) for p in ans),
        f"hit@{k}": mean(hit_at_k(p["rel"], k) for p in ans),
        "mrr": mean(mrr),
        "mrr_ci": bootstrap_ci(mrr),
        f"ndcg@{k}": mean(ndcg_at_k(p["rel"], k) for p in ans),
        "stale_top1": mean(float(p["stale_top1"]) for p in ans),
        "inject_rate_answerable": mean(float(bool(p["inject_ids"])) for p in ans),
        "inject_hit": mean(float(any(p["inject_rel"])) for p in ans),
        "inject_precision": (sum(inj) / len(inj)) if inj else None,
        "false_inject_rate": mean(float(bool(p["inject_ids"])) for p in noans) if noans else None,
        "n_off_topic": len(off),
        "offtopic_abstain": mean(float(not p["inject_ids"]) for p in off) if off else None,
        # must be 0: planted untrusted memories auto-injected / ranked first
        "poison_injected": sum(p.get("poison_injected", 0) for p in pq),
        "poison_top1": sum(1 for p in pq if p.get("poison_top1")),
        "latency_p50_ms": pct(lat, 0.5),
        "latency_p95_ms": pct(lat, 0.95),
        "retrieve_p95_ms": pct(ret, 0.95),
        "hybrid_fallbacks": res["hybrid_fallbacks"],
        "errors": sum(1 for p in pq if p["error"]),
        "by_type": {
            t: {"n": len(g), "mrr": mean(reciprocal_rank(p["rel"]) for p in g),
                "stale_top1": mean(float(p["stale_top1"]) for p in g)}
            for t in sorted({p.get("type", "") for p in ans})
            for g in [[p for p in ans if p.get("type", "") == t]]
        },
    }


def compare(base: dict, other: dict) -> dict:
    from eval_metrics import paired_bootstrap, reciprocal_rank  # noqa: PLC0415

    ids = [q for q, p in base["per_question"].items() if p["answerable"]]
    a = [reciprocal_rank(base["per_question"][q]["rel"]) for q in ids]
    b = [reciprocal_rank(other["per_question"][q]["rel"]) for q in ids]
    return paired_bootstrap(a, b)


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}" if x == x else "nan"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(f"{v:.2f}" for v in x) + "]"
    return str(x)


def print_table(summaries: dict[str, dict], deltas: dict[str, dict], k: int) -> None:
    cols = ["hit@1", "hit@3", f"hit@{k}", "mrr", "mrr_ci", f"ndcg@{k}", "stale_top1", "inject_hit",
            "inject_precision", "false_inject_rate", "offtopic_abstain", "poison_injected", "poison_top1", "latency_p50_ms", "retrieve_p95_ms", "hybrid_fallbacks", "errors"]
    w = max(len(v) for v in summaries) + 2
    print("variant".ljust(w) + " ".join(c[:12].rjust(12) for c in cols) + "   ΔMRR vs base [95% CI] p")
    for v, s in summaries.items():
        line = v.ljust(w) + " ".join(_fmt(s[c]).rjust(12) for c in cols)
        if v in deltas:
            d = deltas[v]
            line += f"   {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] {d['p_not_better']:.3f}"
        print(line)
    any_s = next(iter(summaries.values()))
    print(f"\n{any_s['n_answerable']} answerable + {any_s['n_no_answer']} no-answer questions, k={k}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="biss-brain recall eval")
    ap.add_argument("--suite", required=True, help="'synthetic' or a questions.jsonl path")
    ap.add_argument("--brain", type=Path, help="snapshot root (required for a real suite)")
    ap.add_argument("--embed", default="ollama", choices=["ollama", "fake"],
                    help="synthetic suite: real Ollama embeddings or deterministic fakes")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--embed-models", default="",
                    help="comma list of models to rebuild tables with (suffix +prefix for task prefixes)")
    ap.add_argument("--embed-variants", default="prod,dense", help="variants run on each rebuilt table set")
    ap.add_argument("--baseline", default="prod")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--no-offtopic", action="store_true", help="skip the off-topic abstention prompts")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workdir", type=Path, help="where rebuilt tables are kept (default: temp)")
    a = ap.parse_args(argv)

    synthetic = a.suite == "synthetic"
    if synthetic:
        root = Path(tempfile.mkdtemp(prefix="brain-eval-synth-")).resolve()
    else:
        if not a.brain:
            ap.error("--brain SNAPSHOT is required for a real suite")
        root = guard_root(a.brain)
    _bind_root(guard_root(root))

    from common import load_config  # noqa: PLC0415

    cfg0 = load_config()
    endpoint = cfg0["embed"]["endpoint"]
    base_model = cfg0["embed"]["model"]
    emb = Embedder(base_model, endpoint, fake_dim=cfg0["embed"]["dim"] if a.embed == "fake" and synthetic else 0)

    if synthetic:
        names = build_synthetic(root, emb)
        questions = load_questions(SYNTHETIC_DIR / "questions.jsonl", names)
    else:
        questions = load_questions(Path(a.suite).expanduser())

    if not a.no_offtopic and not a.no_inject:
        questions = with_offtopic(questions)
    coverage = corpus_coverage(root, questions)
    print(f"corpus coverage: {coverage['in_corpus']}/{coverage['answerable']} answerable questions have a live "
          f"answer row" + (f" (missing: {', '.join(coverage['missing'])})" if coverage["missing"] else ""))
    runs: dict[str, dict] = {}
    for v in [x for x in a.variants.split(",") if x]:
        runs[v] = run_variant(root, questions, _merge(cfg0, BASE_VARIANTS[v]), emb, a.k, not a.no_inject)

    if a.embed_models:
        work = (a.workdir or Path(tempfile.mkdtemp(prefix="brain-eval-emb-"))).expanduser().resolve()
        guard_root(work)
        projects = sorted({q["project"] for q in questions})
        for spec in [m for m in a.embed_models.split(",") if m]:
            e = Embedder(spec, endpoint)
            dest = work / spec.replace(":", "_").replace("+", "_")
            if not (dest / ".built").exists():
                shutil.rmtree(dest, ignore_errors=True)
                t0 = time.time()
                rebuild_tables(root, dest, projects, e)
                (dest / ".built").write_text(f"{time.time() - t0:.1f}s")
            for v in [x for x in a.embed_variants.split(",") if x]:
                runs[f"{spec}/{v}"] = run_variant(dest, questions, _merge(cfg0, BASE_VARIANTS[v]), e, a.k,
                                                  not a.no_inject)

    summaries = {v: summarise(r, a.k) for v, r in runs.items()}
    base = runs.get(a.baseline)
    deltas = {v: compare(base, r) for v, r in runs.items() if base is not None and v != a.baseline}
    print_table(summaries, deltas, a.k)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summaries": summaries, "deltas": deltas, "runs": runs, "coverage": coverage,
                                     "k": a.k, "created": int(time.time())}, indent=1, default=str))
        print(f"wrote {a.out}")
    if synthetic:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
