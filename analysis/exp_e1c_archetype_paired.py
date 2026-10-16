#!/usr/bin/env python3
"""E1c — Paired bootstrap for the ARCHETYPE comparison (mask gating, 2000 resamples).

Reviewer ask: attach a paired significance test to the archetype DRAME-Eval vs
best-baseline margin (previously reported as point-estimate only).

The on-disk archetype baseline predictions reproduce the CURRENT masked paper
numbers (Gemini-ZS 0.236, GPT4o-ZS 0.233) — the old "does not reproduce" FLAG was
relative to the stale shipped 0.480, now corrected. DRAME reproduces 0.629 from
ARCHE_DRAME_DIR. Both models are on the same archetype test items, so a paired
Δ(avg) bootstrap on per-dim common ids is well defined.
"""
import json
import os
import numpy as np
import common
import exp_e1_bootstrap as e1

ARCHE_DIMS = common.ARCHE_DIMS


def build_baseline_cells(model, shot, meta):
    preds = common.load_judge_preds_arche(model, shot=shot)
    return e1.build_cells(preds, meta, ARCHE_DIMS, e1.arche_human)


def main():
    os.makedirs(common.OUT_DIR, exist_ok=True)
    meta = common.load_metadata("archetype")

    # DRAME-Eval (reproduces 0.629)
    dp = e1.load_dir_preds(e1.ARCHE_DRAME_DIR, ARCHE_DIMS, is_arche=True)
    cells_drame = e1.build_cells(dp, meta, ARCHE_DIMS, e1.arche_human)
    dp_pt, dlo, dhi = e1.avg_ci(cells_drame, ARCHE_DIMS)
    print(f"DRAME-Eval  archetype Avg = {dp_pt:.3f} [{dlo:.3f}, {dhi:.3f}]")

    baselines = [
        ("Gemini2.5Pro", "zero"),
        ("GPT4o-audio", "zero"),
        ("Gemini2.5Pro", "few"),
        ("GPT4o-audio", "few"),
    ]
    results = {"drame_avg": [dp_pt, dlo, dhi], "baselines": {}, "paired": {}}
    for model, shot in baselines:
        cells_b = build_baseline_cells(model, shot, meta)
        bp, blo, bhi = e1.avg_ci(cells_b, ARCHE_DIMS)
        tag = f"{model} ({'FS' if shot=='few' else 'ZS'})"
        print(f"{tag:22s} archetype Avg = {bp:.3f} [{blo:.3f}, {bhi:.3f}]")
        results["baselines"][tag] = [bp, blo, bhi]
        d, lo, hi, p, ncom = e1.paired_delta_avg(cells_drame, cells_b, ARCHE_DIMS)
        print(f"    paired  DRAME - {tag:22s} Δ={d:+.3f} [{lo:+.3f}, {hi:+.3f}] p(Δ<=0)={p:.4f} n={ncom}")
        results["paired"][tag] = {"delta": d, "ci": [lo, hi], "p_le0": p, "n": ncom}

    json.dump(results, open(os.path.join(common.OUT_DIR, "e1c_archetype_paired.json"), "w"), indent=2)

    # markdown
    md = ["# E1c — Archetype paired bootstrap (mask gating, 2000 resamples)", "",
          f"DRAME-Eval archetype Avg = **{dp_pt:.3f} [{dlo:.3f}, {dhi:.3f}]** (reproduces paper 0.629).", "",
          "| DRAME-Eval − baseline | baseline Avg [95% CI] | Δ Avg | 95% CI | p(Δ≤0) | n |",
          "|---|---|---|---|---|---|"]
    for tag, r in results["paired"].items():
        bp, blo, bhi = results["baselines"][tag]
        md.append(f"| {tag} | {bp:.3f} [{blo:.3f}, {bhi:.3f}] | {r['delta']:+.3f} "
                  f"| [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['p_le0']:.4f} | {r['n']} |")
    md += ["", "Paired Δ uses shared resample indices on per-dimension common ids; "
           "p(Δ≤0) is the fraction of 2,000 resamples with Δ≤0."]
    open(os.path.join(common.OUT_DIR, "e1c_archetype_paired.md"), "w").write("\n".join(md) + "\n")
    print("\nwrote out/e1c_archetype_paired.{json,md}")


if __name__ == "__main__":
    main()
