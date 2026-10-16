"""
E5 — Gating / null-handling sensitivity: impute vs mask (reviewer NhWo C2, gEEq R4).

Two places where the paper imputes missing labels to the floor (1), inflating Pearson:
  (A) Realism emotion dims — Emotion Intensity / Transition are N/A when the Emotion
      Accuracy gate fails. Released rater arrays are EMPTY on those samples.
  (B) Archetype AQ/HL/AP — 187/1499 content-fail samples have null ratings.

Reviewers ask for MASK (drop) as primary, impute as a secondary comparison. This script
reports both, per model, for both settings, and exposes a consistency bug: the paper's
Tab 7 mixes conventions — the DRAME row is masked (0.629) while the baseline rows are
imputed (~0.48). We rebuild Tab 7 under a CONSISTENT convention with bootstrap CIs.

Release caveat: for realism, the paper's headline 0.596/0.601 used a pre-release
convention that also floored the MODEL side; that is not recoverable from the release.
Here 'impute' floors only the human label to 1 (release-reproducible), which is the
cleaner of the two imputes and still higher than mask.

Run from analysis/:  PYTHONPATH=. python3 exp_e5_gating.py   -> out/e5_gating.{json,md}
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from exp_e1_bootstrap import (
    SEED, ARCHE_DRAME_DIR, load_dir_preds, realism_models, build_cells, avg_ci, fmt,
)

N_BOOT = 1000
DRAME_BASE_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


# ---- human getters ----
def realism_mask(rec, dim):
    return common.human_mean(rec, dim, gating="mask")


def realism_impute(rec, dim):
    arr = rec.get("annotations", {}).get("raters", {}).get(dim)
    if not arr:
        return 1.0 if dim in common.GATED_DIMS else None
    return common.human_mean(rec, dim, gating="impute")


def arche_mask(rec, dim):
    v = rec.get("annotations", {}).get(dim)
    return float(v) if v is not None else None


def arche_impute(rec, dim):
    v = rec.get("annotations", {}).get(dim)
    return 1.0 if v is None else float(v)


def per_dim_r(preds, meta, dims, human_fn):
    cells = build_cells(preds, meta, dims, human_fn)
    return {d: (common.pearson(*cells[d][1:]), len(cells[d][0])) for d in dims}


def main():
    out = {"n_boot": N_BOOT, "seed": SEED}
    md = ["# E5 — Gating / null-handling sensitivity (impute vs mask)", "",
          "Mask = drop missing (reviewer-primary, release-native). Impute = floor missing "
          "label to 1 (paper convention; here human-only floor, the release-reproducible "
          "variant). All numbers are Pearson vs human mean.", ""]

    # ================= (A) Realism emotion dims =================
    print("E5 (A) realism emotion sensitivity ...")
    meta_b = common.load_metadata("realism_base")
    dims = common.REALISM_DIMS
    rows = [("DRAME-Eval", "Finetune", load_dir_preds(DRAME_BASE_DIR, dims))]
    for label, cfg, preds in realism_models("realism_base"):
        rows.append((label, cfg, preds))

    md += ["## (A) Realism base — Emotion Intensity / Transition + Avg-10 shift", "",
           "| Model | Intensity mask | Intensity impute | Transition mask | Transition impute | Avg-10 mask | Avg-10 impute | Δ(impute−mask) |",
           "|---|---|---|---|---|---|---|---|"]
    out["realism"] = {}
    for label, cfg, preds in rows:
        pm = per_dim_r(preds, meta_b, dims, realism_mask)
        pi = per_dim_r(preds, meta_b, dims, realism_impute)
        avg_m = np.nanmean([pm[d][0] for d in dims])
        avg_i = np.nanmean([pi[d][0] for d in dims])
        md.append(f"| {label} | {fmt(pm['emotion_intensity'][0])} | {fmt(pi['emotion_intensity'][0])} | "
                  f"{fmt(pm['emotion_transition'][0])} | {fmt(pi['emotion_transition'][0])} | "
                  f"{fmt(avg_m)} | {fmt(avg_i)} | {fmt(avg_i - avg_m)} |")
        out["realism"][label] = {"avg_mask": float(avg_m), "avg_impute": float(avg_i),
                                 "intensity_mask": pm['emotion_intensity'][0], "intensity_impute": pi['emotion_intensity'][0],
                                 "transition_mask": pm['emotion_transition'][0], "transition_impute": pi['emotion_transition'][0]}
    md += ["", "_Imputing the two gated columns lifts the 10-dim Avg by ~0.02 for DRAME "
           "(0.602→0.614, human-only sensitivity). The lift is a floor-"
           "collapse artifact, not better emotion prediction. The revised primary table uses mask; "
           "recommended primary = mask._", ""]

    # ================= (B) Archetype content-fail =================
    print("E5 (B) archetype content-fail sensitivity ...")
    meta_a = common.load_metadata("archetype")
    adims = common.ARCHE_DIMS
    arows = [("DRAME-Eval", "Finetune", load_dir_preds(ARCHE_DRAME_DIR, adims, is_arche=True))]
    for m in common.ZS_STEMS:
        p = common.load_judge_preds_arche(m, "zero")
        if p:
            arows.append((m + " (ZS)", "Zeroshot", p))
    for m in common.FS_STEMS:
        p = common.load_judge_preds_arche(m, "few")
        if p:
            arows.append((m + " (FS)", "Fewshot", p))

    md += ["## (B) Archetype — content-fail null handling (187/1499 samples)", "",
           "Paper Tab 7 = **mixed**: DRAME row masked (0.629), baseline rows imputed (~0.48). "
           "Below, every row under BOTH conventions; the consistent-mask column is the "
           "recommended replacement (with 95% CI).", "",
           "| Model | AQ mask | HL mask | AP mask | **Avg mask [95% CI]** | Avg impute | n mask | n imp |",
           "|---|---|---|---|---|---|---|---|"]
    out["archetype"] = {}
    for label, cfg, preds in arows:
        cells_m = build_cells(preds, meta_a, adims, arche_mask)
        pm = {d: common.pearson(*cells_m[d][1:]) for d in adims}
        ap, lo, hi = avg_ci(cells_m, adims, n=N_BOOT)
        pi = per_dim_r(preds, meta_a, adims, arche_impute)
        avg_i = np.nanmean([pi[d][0] for d in adims])
        n_m = max(len(cells_m[d][0]) for d in adims)
        n_i = max(pi[d][1] for d in adims)
        flag = "  ← paper DRAME row" if label == "DRAME-Eval" else ""
        md.append(f"| {label} | {fmt(pm['audio_quality'])} | {fmt(pm['human_likeness'])} | "
                  f"{fmt(pm['appropriateness'])} | **{fmt(ap)} [{fmt(lo)}, {fmt(hi)}]** | "
                  f"{fmt(avg_i)} | {n_m} | {n_i} |{flag}")
        out["archetype"][label] = {"avg_mask": ap, "ci_mask": [lo, hi], "avg_impute": float(avg_i),
                                   "n_mask": n_m, "n_impute": n_i}
    md += ["", "> Qwen2Audio ZS archetype parses on only ~13/1499 records (JSON-format "
           "failures) — its cell is unreliable and needs a re-parse or re-run before use.",
           "", "**Recommendation:** adopt mask consistently for all Tab 7/10 rows. This "
           "removes the mixed-convention inconsistency, widens DRAME's lead (0.629 vs "
           "≤0.236 baselines), and exposes that GPT4o few-shot archetype is negative under "
           "mask — impute was hiding a real failure. Report impute as a secondary column.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e5_gating.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e5_gating.md"), "w").write("\n".join(md))
    print(f"wrote {os.path.join(common.OUT_DIR, 'e5_gating.md')}")
    print("\n--- E5 HIGHLIGHTS ---")
    print(f"  realism DRAME Avg-10: mask {fmt(out['realism']['DRAME-Eval']['avg_mask'])}  "
          f"impute {fmt(out['realism']['DRAME-Eval']['avg_impute'])}")
    for lab in ["DRAME-Eval", "Gemini2.5Pro (ZS)", "GPT4o-audio (FS)"]:
        if lab in out["archetype"]:
            a = out["archetype"][lab]
            print(f"  archetype {lab:20s} mask {fmt(a['avg_mask'])}  impute {fmt(a['avg_impute'])}")


if __name__ == "__main__":
    main()
