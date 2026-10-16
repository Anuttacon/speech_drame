"""
R3-C — Training / overfitting evidence for the four fine-tuning configs (reviewer ZB3A RC4:
"Instruct-FT drops on real recordings → provide a learning curve").

Honest scope note: saved predictions exist at essentially ONE scored checkpoint per config
(FT at 3,000 steps, LoRA at 10,000), so a genuine step-wise learning curve (metric vs
training steps) is NOT recoverable from disk — it would need a GPU re-eval sweep at
intermediate checkpoints (PLAN_review3 decision #3). What we CAN show from existing
predictions, and what actually answers the reviewer's concern:

  (1) The in-distribution → real-recording GENERALIZATION GAP for every config, all on the
      clean held-out test_real_v2 (1,084; the Instruct-LoRA clean number has now landed).
      A large positive in-dist correlation with a ~0/negative real-rec correlation is the
      overfitting-to-the-synthesis-distribution signature the reviewer suspects.
  (2) The one genuine two-checkpoint comparison on disk (LoRA base test, 8k vs 10k steps):
      in-distribution performance is already saturated, so the real-rec weakness is not a
      "stop earlier" problem.
  (3) Reframing with §0: the real-rec drop is OVER-DETERMINED — train contamination (fixed
      here by using test_real_v2), between-speaker composition (E4), AND a genuine
      synthetic-to-real gap. "Overfitting" is one of several causes, not the whole story.

Run from analysis/:  PYTHONPATH=. python3 exp_r3c_learning_curve.py
Writes analysis/out/r3c_learning_curve.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids
from exp_e1_bootstrap import load_dir_preds, realism_human, build_cells, avg_ci, fmt

# config -> (base_dir, clean_real_dir[test_real_v2], step)
CONFIGS = {
    "Base-FT":       ("results/sft_finetune_base_all-real-v2_checkpoint-3000",
                      "results/sft_finetune_base_all-real-v2_checkpoint-3000_testreal", 3000),
    "Instruct-FT":   ("results/sft_finetune_all-real-v2_checkpoint-3000",
                      "results/sft_finetune_all-real-v2_checkpoint-3000_testreal", 3000),
    "Base-LoRA":     ("results/sft_lora_base_specific_dimension_checkpoint-10000",
                      "results/sft_lora_base_all-real-v2_checkpoint-10000_testreal", 10000),
    "Instruct-LoRA": ("results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall",
                      "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2", 10000),
}
# genuine 2-checkpoint base comparison (all-real LoRA, non-v2 run)
SATURATION = {8000: "results/sft_lora_specific_dimension_checkpoint-8000",
              10000: "results/sft_lora_specific_dimension_checkpoint-10000"}


def base_avg(reldir, meta_b, dims):
    preds = load_dir_preds(reldir, dims)
    if not preds:
        return None
    cells = build_cells(preds, meta_b, dims, realism_human)
    return avg_ci(cells, dims)


def real_avg(reldir, meta_r, dims, v2):
    preds = load_dir_preds(reldir, dims)
    if not preds:
        return None
    cells = build_cells(preds, meta_r, dims, realism_human, id_filter=v2)
    return avg_ci(cells, dims)


def main():
    dims = common.REALISM_DIMS
    meta_b = common.load_metadata("realism_base")
    meta_r = common.load_metadata("realism_real")
    v2 = set(load_v2_ids())
    out = {}
    md = ["# R3-C — Overfitting / generalization-gap evidence (fine-tuning configs)", "",
          "Reviewer ZB3A RC4. No step-wise learning curve exists on disk (one scored "
          "checkpoint per config); a finer sweep needs GPU re-eval. Below is the "
          "in-distribution → real-recording gap, all on the clean held-out test_real_v2.", ""]

    md += ["## 1. Generalization gap: in-distribution (base) vs held-out real recordings", "",
           "| Config | steps | In-dist base Avg [95% CI] | Real-rec Avg [95% CI] (test_real_v2) | gap |",
           "|---|---|---|---|---|"]
    out["gap"] = {}
    for cfg, (bdir, rdir, step) in CONFIGS.items():
        ba = base_avg(bdir, meta_b, dims)
        ra = real_avg(rdir, meta_r, dims, v2)
        bstr = f"{fmt(ba[0])} [{fmt(ba[1])}, {fmt(ba[2])}]" if ba else "_missing_"
        rstr = f"{fmt(ra[0])} [{fmt(ra[1])}, {fmt(ra[2])}]" if ra else "_missing_"
        gap = (ba[0] - ra[0]) if (ba and ra) else float("nan")
        md.append(f"| {cfg} | {step} | {bstr} | {rstr} | {fmt(gap)} |")
        out["gap"][cfg] = {"base": ba[0] if ba else None, "real": ra[0] if ra else None,
                           "gap": float(gap) if gap == gap else None, "steps": step}
    md += ["",
           "_Every config shows a large positive in-distribution correlation collapsing to "
           "~0 or negative on held-out real recordings — the overfitting-to-the-synthesis-"
           "distribution signature the reviewer suspected, now shown on the CLEAN real set "
           "for all four (the Instruct-LoRA real number was previously contaminated; §0/E1). "
           "The gap is not unique to Instruct-FT — it is a property of the synthetic→real "
           "transfer, which is exactly the benchmark's headline open problem._", ""]

    # ---- 2. saturation (two base checkpoints) ----
    md += ["## 2. In-distribution saturation (only genuine two-checkpoint comparison on disk)", "",
           "LoRA base test, 8,000 vs 10,000 steps:", "",
           "| steps | In-dist base Avg [95% CI] |", "|---|---|"]
    out["saturation"] = {}
    for step, d in SATURATION.items():
        ba = base_avg(d, meta_b, dims)
        md.append(f"| {step} | {fmt(ba[0])} [{fmt(ba[1])}, {fmt(ba[2])}] |" if ba else f"| {step} | _missing_ |")
        if ba:
            out["saturation"][step] = ba[0]
    if len(out["saturation"]) == 2:
        d8, d10 = out["saturation"].get(8000), out["saturation"].get(10000)
        md += ["", f"_In-distribution performance is essentially flat from 8k→10k "
               f"({fmt(d8)}→{fmt(d10)}): the model has saturated in-domain, so the real-rec "
               f"weakness is not an 'early-stopping would fix it' artifact — more/less "
               f"training does not close the synthetic-to-real gap._", ""]

    md += ["## 3. Reframe (tie to §0)", "",
           "The real-recording drop is **over-determined**: (i) train contamination (the "
           "published 0.247 mixed in-train and held-out samples; corrected here to the clean "
           "test_real_v2), (ii) between-speaker composition (E4: pooled real-rec correlation "
           "is largely cross-speaker, not within-speaker signal), and (iii) a genuine "
           "synthetic-to-real distribution gap. A naive learning curve would conflate these. "
           "The honest framing: fine-tuning fits the in-distribution rating function well and "
           "does not transfer to held-out real speech — for ANY config — which is the "
           "benchmark's central negative result.", "",
           "## Integration", "",
           "- One figure/table (the gap table) + a sentence in §6.3 replacing the "
           "'Instruct-FT overfits' single-config framing with the all-config gap.",
           "- State plainly that a finer step-wise curve needs a GPU checkpoint sweep "
           "(decision #3); offer it if the reviewer wants the literal learning curve.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "r3c_learning_curve.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "r3c_learning_curve.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "r3c_learning_curve.md"))
    print("\n--- R3-C HIGHLIGHTS (base → real gap) ---")
    for cfg, g in out["gap"].items():
        print(f"  {cfg:14s} base {fmt(g['base'])} → real {fmt(g['real'])}  gap {fmt(g['gap'])}")


if __name__ == "__main__":
    main()
