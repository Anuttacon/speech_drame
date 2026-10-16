"""
E14 (model side) — DRAME-Eval vs human 13-system ranking on the annotated benchmark
(reviewer gEEq; intended-use payoff). Run AFTER the benchmark inference job (job-nh8l208m0c)
lands its predictions in results/benchmark_e14_drame/.

Coverage: 11/13 systems (both vevo excluded — no local wavs). Model per-system score = mean
over the 10 predicted dims; human per-system score = mean over 11 annotated dims (from the
release benchmark.jsonl). Report Spearman/Kendall of the model-vs-human system ranking
(system-level, n=11) + per-dim rank agreement.

Run from analysis/:  PYTHONPATH=. python3 exp_e14_model.py
Writes analysis/out/e14_model.{json,md}  (complements e14_benchmark_ranking.md, the human side).
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from exp_e1_bootstrap import load_dir_preds

PRED_DIR = "results/benchmark_e14_drame"
BENCH = os.path.join(common.REALISM_META, "benchmark.jsonl")
ALL_DIMS = common.REALISM_DIMS + ["semantic_match"]


def human_sample_avg(rec):
    vals = []
    for d in ALL_DIMS:
        if d in common.REALISM_DIMS:
            v = common.human_mean(rec, d, "mask")
        else:
            arr = rec.get("annotations", {}).get("raters", {}).get("semantic_match") or []
            arr = [float(x) for x in arr if x is not None]
            v = float(np.mean(arr)) if arr else None
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def main():
    if not os.path.exists(os.path.join(common.EVAL_DIR, PRED_DIR, "role_eval_pitch_dynamics_exact.json")):
        print("Benchmark model preds not present yet — waiting on job-nh8l208m0c. Re-run when it lands.")
        return
    rel = [json.loads(l) for l in open(BENCH)]
    rel_by_id = {common._norm_id(r["id"]): r for r in rel}  # strip .wav to match pred ids
    preds = load_dir_preds(PRED_DIR, common.REALISM_DIMS)  # {id -> {dim: score}}

    # per-sample model avg (over 10 dims) + human avg, grouped by system
    by_sys_model = collections.defaultdict(list)
    by_sys_human = collections.defaultdict(list)
    for rid, pr in preds.items():
        rec = rel_by_id.get(rid)
        if rec is None:
            continue
        s = rec["tts"]["system"]
        m = np.mean([pr[d] for d in common.REALISM_DIMS if d in pr])
        h = human_sample_avg(rec)
        if h is not None:
            by_sys_model[s].append(m); by_sys_human[s].append(h)

    systems = sorted(by_sys_model, key=lambda s: -np.mean(by_sys_human[s]))
    rows = [(s, float(np.mean(by_sys_human[s])), float(np.mean(by_sys_model[s])), len(by_sys_model[s]))
            for s in systems]
    hvec = [r[1] for r in rows]; mvec = [r[2] for r in rows]
    rho = common.spearman(hvec, mvec); tau = common.kendall(hvec, mvec); r = common.pearson(hvec, mvec)

    md = ["# E14 (model side) — DRAME-Eval vs human system ranking (benchmark)", "",
          f"11/13 systems (vevo excluded, no local wavs). System-level agreement over "
          f"n={len(rows)} systems. Model score = mean of 10 predicted dims; human = mean of "
          f"11 annotated dims.", "",
          f"**Spearman ρ = {rho:+.3f}   Kendall τ = {tau:+.3f}   Pearson r = {r:+.3f}** "
          f"(system ranking, n={len(rows)}).", "",
          "| System | human mean | model mean | n | human rank | model rank |",
          "|---|---|---|---|---|---|"]
    # ranks (1 = highest score)
    hrank = {s: i + 1 for i, (s, _, _, _) in enumerate(sorted(rows, key=lambda x: -x[1]))}
    mrank = {s: i + 1 for i, (s, _, _, _) in enumerate(sorted(rows, key=lambda x: -x[2]))}
    for s, h, m, n in rows:
        md.append(f"| {s} | {h:.3f} | {m:.3f} | {n} | {hrank[s]} | {mrank[s]} |")
    md += ["",
           f"_DRAME-Eval's system ranking {'agrees with' if rho>0.5 else 'partially tracks' if rho>0 else 'does not track'} "
           f"the human ranking (ρ={rho:+.3f}). Caveats (state in text): ZH-only, ~20 "
           f"scenes/system, 11/13 systems; scenes/voices are from training corpora (constant "
           f"across systems, cannot drive the ranking); 3 TTS backbones (CosyVoice2/MaskGCT/"
           f"Vevo) were in the training bank — group backbone-seen vs unseen and verify "
           f"agreement is not explained by that; cross-check vs a generic MOS predictor "
           f"(E17)._", ""]

    out = {"spearman": rho, "kendall": tau, "pearson": r, "n_systems": len(rows),
           "systems": [{"system": s, "human": h, "model": m, "n": n} for s, h, m, n in rows]}
    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e14_model.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e14_model.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e14_model.md"))
    print(f"  system ranking: Spearman {rho:+.3f}  Kendall {tau:+.3f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
