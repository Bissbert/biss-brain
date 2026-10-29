#!/usr/bin/env python3
"""Fit the UserPromptSubmit injection gate on eval runs.

  calibrate_inject.py RUN.json [RUN.json ...] [--variant prod] [--novec-variant novec]

Each RUN.json is a run_eval.py --out file. The gate is replayed through
retrieve.inject_gate on the recorded pre-gate candidates, so the fit is the
shipped code path. A setting qualifies when, on every suite, item precision
>= --min-precision, off-topic abstention >= --min-abstain, no-answer false
injection <= --max-false and no poisoned item is injected; among those the
one with the best mean inject_hit (answerable prompts that got a relevant
item) wins.

The fit is in-sample: it picks the best of a few hundred settings on the
suites it then reports, so precision carries a Wilson 95 % interval and a
cross-suite check (fit on one suite, score the other) is printed too.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrieve  # noqa: E402
from common import DEFAULT_CONFIG, _deep_merge  # noqa: E402


def replay(per_q: dict, gate: dict, vector: bool) -> dict:
    cfg = _deep_merge(DEFAULT_CONFIG, {"retrieve": gate})
    inj, hits, false, off, poison = [], [], [], [], 0
    for p in per_q.values():
        rows = [{"_cosine": c["cos"], "_bm25": c["bm25"], "text": "x" * c["chars"], "_rel": c["rel"],
                 "_poison": c["poison"]} for c in p.get("inject_cands", [])]
        got = retrieve.inject_gate(rows, cfg, vector=vector)
        poison += sum(1 for r in got if r["_poison"])
        if p["answerable"]:
            inj += [r["_rel"] for r in got]
            hits.append(float(any(r["_rel"] for r in got)))
        elif p.get("type") == "off_topic":
            off.append(float(not got))
        else:
            false.append(float(bool(got)))

    def m(xs):
        return sum(xs) / len(xs) if xs else None

    return {"precision": m(inj), "inject_hit": m(hits), "false_inject": m(false), "offtopic_abstain": m(off),
            "poison": poison, "n_injected": len(inj), "precision_ci": wilson(sum(inj), len(inj))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 2), round(c + h, 2))


def ok(r: dict, a) -> bool:
    return (r["poison"] == 0 and (r["precision"] is None or r["precision"] >= a.min_precision)
            and (r["offtopic_abstain"] is None or r["offtopic_abstain"] >= a.min_abstain)
            and (r["false_inject"] is None or r["false_inject"] <= a.max_false))


def fit(suites: dict[str, dict], grid: list[dict], vector: bool, a) -> list[tuple]:
    out = []
    for gate in grid:
        res = {name: replay(pq, gate, vector) for name, pq in suites.items()}
        score = sum(r["inject_hit"] or 0.0 for r in res.values()) / len(res)
        out.append((all(ok(r, a) for r in res.values()), score, gate, res))
    out.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--variant", default="prod")
    ap.add_argument("--novec-variant", default="novec")
    ap.add_argument("--min-precision", type=float, default=0.7)
    ap.add_argument("--min-abstain", type=float, default=0.8)
    ap.add_argument("--max-false", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    data = {p.stem: json.loads(p.read_text())["runs"] for p in a.runs}

    vec_grid = [
        {"inject_min_cosine": c, "inject_min_bm25": b, "inject_top_k": k}
        for c, b, k in itertools.product(
            [x / 100 for x in range(50, 82, 2)] + [2.0], [0, 6, 7, 8, 9, 10, 11, 12, 14], [1, 2, 3])
    ]
    bm_grid = [{"inject_min_bm25": b, "inject_top_k": k} for b, k in itertools.product(
        [0, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20], [1, 2, 3])]
    for label, variant, grid, vector in (("vector", a.variant, vec_grid, True),
                                          ("no-vector", a.novec_variant, bm_grid, False)):
        suites = {n: r[variant]["per_question"] for n, r in data.items() if variant in r}
        if not suites:
            continue
        print(f"\n== {label} gate ({variant}); qualifies = poison 0, precision >= {a.min_precision}, "
              f"off-topic abstain >= {a.min_abstain}, false inject <= {a.max_false} on every suite")
        for good, score, gate, res in fit(suites, grid, vector, a)[: a.top]:
            print(f"{'OK ' if good else '-- '} {gate}  mean_hit={score:.3f}\n      {_cells(res)}")
        if len(suites) > 1:
            print("  cross-suite: best qualifying gate fit on one suite, scored on the others")
            for name in suites:
                best = fit({name: suites[name]}, grid, vector, a)[0]
                rest = {n: replay(pq, best[2], vector) for n, pq in suites.items() if n != name}
                print(f"    fit {name}: {best[2]} -> {_cells(rest)}")
    return 0


def _cells(res: dict) -> str:
    return "  ".join(
        f"{n}: P={_f(r['precision'])} {r['precision_ci'] or ''} hit={_f(r['inject_hit'])} "
        f"F={_f(r['false_inject'])} abst={_f(r['offtopic_abstain'])} n={r['n_injected']}" for n, r in res.items())


def _f(x) -> str:
    return "  - " if x is None else f"{x:.2f}"


if __name__ == "__main__":
    sys.exit(main())
