"""
E14 (human side) — 13-system realism ranking on the annotated benchmark (reviewer gEEq;
the intended-use payoff = using DRAME-Eval to rank TTS/speech systems).

The benchmark split (260 = 13 systems x 20 scenes, ZH, all delivery-negatives) carries
per-sample `tts.system` and rater arrays. Here we compute the HUMAN per-system ranking
(mean over the 11 annotated dimensions, mask gating) with scene-level bootstrap CIs.

The MODEL side (DRAME-Eval predictions on the 260) is not on disk — 260-sample inference on
the cluster is trivial but pending; once it lands, add per-dim Spearman/Kendall(model human
system-ranking) + scene-level paired CI. Caveats to state (per PLAN_review2 E14): ZH-only,
20 scenes/system; scenes/voices come from training corpora (constant across the 13 systems,
so cannot drive the ranking); 3 TTS backbones were in the training bank (group seen vs
unseen backbone and verify agreement is not explained by that).

Run from analysis/:  PYTHONPATH=. python3 exp_e14_benchmark_ranking.py
Writes analysis/out/e14_benchmark_ranking.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common

BENCH = os.path.join(common.REALISM_META, "benchmark.jsonl")
ALL_DIMS = common.REALISM_DIMS + ["semantic_match"]


def load():
    return [json.loads(l) for l in open(BENCH)]


def sample_human_avg(rec):
    """Mean over the 11 dims of the rater-mean (mask gating)."""
    vals = []
    for d in ALL_DIMS:
        v = common.human_mean(rec, d, "mask") if d in common.REALISM_DIMS else None
        if d == "semantic_match":
            arr = rec.get("annotations", {}).get("raters", {}).get("semantic_match") or []
            arr = [float(x) for x in arr if x is not None]
            v = float(np.mean(arr)) if arr else None
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def main():
    recs = load()
    by_sys = collections.defaultdict(list)
    for r in recs:
        s = r.get("tts", {}).get("system")
        a = sample_human_avg(r)
        if s and a is not None:
            by_sys[s].append(a)
    # rank systems by mean human overall, scene-level bootstrap CI
    rng = np.random.default_rng(12345)
    rows = []
    for s, vals in by_sys.items():
        vals = np.array(vals)
        boot = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(2000)]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append((s, float(vals.mean()), float(lo), float(hi), len(vals)))
    rows.sort(key=lambda x: -x[1])

    md = ["# E14 (human side) — 13-system benchmark ranking", "",
          f"Human per-system quality on the annotated benchmark ({len(recs)} = "
          f"{len(by_sys)} systems x ~20 scenes, ZH). Mean over 11 dims (mask), scene-level "
          f"bootstrap 95% CI. **Model side (DRAME-Eval on the 260) pending 260-sample "
          f"inference** — rank correlation lands then.", "",
          "| Rank | System | Human mean [95% CI] | n scenes |", "|---|---|---|---|"]
    out = {"n_systems": len(by_sys), "ranking": []}
    for i, (s, m, lo, hi, n) in enumerate(rows, 1):
        md.append(f"| {i} | {s} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {n} |")
        out["ranking"].append({"rank": i, "system": s, "human_mean": m, "ci": [lo, hi], "n": n})
    spread = rows[0][1] - rows[-1][1]
    md += ["",
           f"_Human quality spans {rows[-1][1]:.2f}-{rows[0][1]:.2f} across the 13 systems "
           f"(spread {spread:.2f}) — a real ranking signal for the model to reproduce. "
           f"Best: {rows[0][0]}; worst: {rows[-1][0]}._", "",
           "## Pending (model side) + caveats", "",
           "- Run DRAME-Eval on the 260 (cluster, trivial) → add per-dim Spearman/Kendall of "
           "the model-vs-human system ranking + scene-level paired bootstrap CI.",
           "- Caveats to state: ZH-only, ~20 scenes/system, small sample-level n; scenes/"
           "voices are from training corpora (constant across systems → cannot drive the "
           "ranking); group the 13 by training-bank-seen vs unseen TTS backbone and verify "
           "agreement is not explained by that grouping; cross-check vs a generic MOS "
           "predictor (E17) to show signal beyond naturalness.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e14_benchmark_ranking.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e14_benchmark_ranking.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e14_benchmark_ranking.md"))
    print("\n--- E14 (human) HIGHLIGHTS ---")
    print(f"  {len(by_sys)} systems; human mean range {rows[-1][1]:.2f}-{rows[0][1]:.2f}")
    for s, m, lo, hi, n in rows[:3]:
        print(f"  top: {s} {m:.3f}")


if __name__ == "__main__":
    main()
