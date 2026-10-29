#!/usr/bin/env python3
"""Replay ablation for the stateful read-side mechanisms (v0.7 phase 7).

  ablate.py --suite synthetic [--embed fake|ollama] [--out F]
  ablate.py --suite QUESTIONS.jsonl --brain SNAPSHOT [--out F]

run_eval.py measures a frozen store. The post-retrieval effects (retrieval
counts, STDP edge bumps, RIF) change the store, and dream's decay pass then
rescores it, so they only show up over a sequence of reads. This replays the
suite against private copies of the snapshot:

  frozen        no reads recorded
  frozen_dream  frozen, then dream stage 1c (decay recomputed for "now")
  use           every answerable question read once with all effects on
  use_norif     the same reads with RIF off
  use_dream     use, then dream stage 1c

and measures every arm with the production ranking (no injection). Pairs:
  RIF before the nightly pass   use vs use_norif
  usage feedback through decay  use_dream vs frozen_dream

The replay reads the questions it then measures, so the usage numbers are an
in-sample upper bound, not a forecast. It also times apply_post_retrieval
per read with each effect on and off. Output: ids and numbers only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval as E  # noqa: E402

ARMS = ["frozen", "frozen_dream", "use", "use_norif", "use_dream"]
PAIRS = {"rif_live": ("use_norif", "use"), "usage_via_decay": ("frozen_dream", "use_dream")}


def _replay(arm_root: Path, questions: list[dict], cfg: dict, emb, k: int) -> list[float]:
    """Read each answerable question once and record the effects; returns ms per effects call."""
    import sqlite3  # noqa: PLC0415

    import effects  # noqa: PLC0415
    from eval_metrics import is_answerable  # noqa: PLC0415
    import retrieve  # noqa: PLC0415

    ms = []
    conns: dict[str, sqlite3.Connection] = {}
    for q in questions:
        if not is_answerable(q):
            continue
        proj = q["project"]
        tbl = E.open_table(arm_root, proj)
        if tbl is None:
            continue
        conn = conns.get(proj) or conns.setdefault(proj, sqlite3.connect(str(arm_root / proj / "meta.sqlite")))
        qvec = emb.embed([q["query"]], "query")[0]
        rows = retrieve.retrieve(tbl, conn, proj, q["query"], qvec, cfg, top_k=k)
        t = time.perf_counter()
        effects.apply_post_retrieval(conn, tbl, proj, rows, cfg, "ablate")
        ms.append((time.perf_counter() - t) * 1000)
    for c in conns.values():
        c.close()
    return ms


def _dream_decay(arm_root: Path, cfg: dict) -> None:
    import dream  # noqa: PLC0415

    for pdir in sorted(p for p in arm_root.iterdir() if (p / "meta.sqlite").is_file()):
        dream._stage1c_decay(pdir, cfg, dry_run=False)


def _copy(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("config.toml", "SNAPSHOT.json", "*.pid", "*.log"))
    return dest


def _pct(xs: list[float], f: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--brain", type=Path)
    ap.add_argument("--embed", default="ollama", choices=["ollama", "fake"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    base = Path(tempfile.mkdtemp(prefix="brain-ablate-")).resolve()
    synthetic = a.suite == "synthetic"
    if not synthetic:
        if not a.brain:
            ap.error("--brain SNAPSHOT is required for a real suite")
        src = E.guard_root(a.brain)
        if (src / "config.toml").exists():
            shutil.copy2(src / "config.toml", base / "config.toml")
    E._bind_root(E.guard_root(base))

    from common import _deep_merge, load_config  # noqa: PLC0415

    cfg = load_config()
    emb = E.Embedder(cfg["embed"]["model"], cfg["embed"]["endpoint"],
                     fake_dim=cfg["embed"]["dim"] if a.embed == "fake" and synthetic else 0)
    if synthetic:
        src = base / "_src"
        src.mkdir()
        names = E.build_synthetic(src, emb)
        questions = E.load_questions(E.SYNTHETIC_DIR / "questions.jsonl", names)
    else:
        questions = E.load_questions(Path(a.suite).expanduser())

    norif = _deep_merge(cfg, {"rif": {"enabled": False}})
    runs, timing = {}, {}
    for arm in ARMS:
        root = _copy(src, base / arm)
        if arm.startswith("use"):
            timing[arm] = _replay(root, questions, norif if arm == "use_norif" else cfg, emb, a.k)
        if arm.endswith("_dream"):
            _dream_decay(root, cfg)
        runs[arm] = E.run_variant(root, questions, cfg, emb, a.k, inject=False)

    # effect cost per read, on fresh copies so the counts do not compound
    cost = {}
    for label, over in {"all": {}, "no_rif": {"rif": {"enabled": False}},
                        "no_stdp": {"stdp": {"enabled": False}},
                        "neither": {"rif": {"enabled": False}, "stdp": {"enabled": False}}}.items():
        root = _copy(src, base / f"cost_{label}")
        ms = _replay(root, questions, _deep_merge(cfg, over), emb, a.k)
        cost[label] = {"p50_ms": _pct(ms, 0.5), "p95_ms": _pct(ms, 0.95), "mean_ms": statistics.fmean(ms) if ms else 0}

    summaries = {arm: E.summarise(r, a.k) for arm, r in runs.items()}
    deltas = {name: E.compare(runs[b], runs[x]) for name, (b, x) in PAIRS.items()}
    for arm, s in summaries.items():
        print(f"{arm:14s} mrr {s['mrr']:.3f} hit@{a.k} {s[f'hit@{a.k}']:.3f}")
    for name, d in deltas.items():
        print(f"{name:16s} ΔMRR {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]")
    for label, c in cost.items():
        print(f"effects {label:8s} p50 {c['p50_ms']:.1f} ms  p95 {c['p95_ms']:.1f} ms")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summaries": summaries, "deltas": deltas, "effects_cost": cost,
                                     "k": a.k, "created": int(time.time())}, indent=1, default=str))
        print(f"wrote {a.out}")
    shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
