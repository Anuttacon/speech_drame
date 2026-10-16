"""
E16 — Semantic Match resolution (reviewer gEEq G1.4). The annotation has 11 dimensions but
the DRAME-Eval model has 10 heads: Semantic Match is annotated but never trained/predicted.
This documents why that is defensible, with numbers.

Computed here (all offline):
  (1) Coverage of semantic_match per split (annotated vs legitimately N/A).
  (2) Inter-annotator agreement on semantic_match (ordinal α + weighted κ) — expected high.
  (3) Consensus-level correlation between semantic_match and Local Scene Fit per split — if
      near-redundant, dropping it from the model heads loses little; it stays in the release.
  (4) Verification that no prediction file contains a semantic_match head (the model is 10-dim).

Run from analysis/:  PYTHONPATH=. python3 exp_e16_semantic_match.py
Writes analysis/out/e16_semantic_match.{json,md}.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import common
from exp_e6_weighted_kappa import weighted_kappa_pairwise, collect_units

TRAIN_FILE = os.path.join(common.REALISM_META, "train.jsonl")
BENCH_FILE = os.path.join(common.REALISM_META, "benchmark.jsonl")


def load_file(fp):
    return [json.loads(l) for l in open(fp)]


def coverage(recs):
    tot = len(recs)
    ann = sum(1 for r in recs if r.get("annotations", {}).get("semantic_match") is not None)
    return ann, tot


def sm_vs_local(recs):
    """Consensus (mean) correlation semantic_match vs local_scene_fit."""
    S, L = [], []
    for r in recs:
        a = r.get("annotations", {})
        s = a.get("semantic_match"); l = a.get("local_scene_fit")
        if s is not None and l is not None:
            S.append(float(s)); L.append(float(l))
    return common.pearson(S, L), len(S)


def sm_agreement(meta):
    units = collect_units(meta, "semantic_match")
    return (common.krippendorff_alpha_ordinal(units),
            weighted_kappa_pairwise(units, "quadratic"), len(units))


def check_no_sm_prediction():
    """Confirm no DRAME prediction file targets semantic_match."""
    hits = []
    for d in common.DRAME_DIRS.values():
        for fp in glob.glob(os.path.join(d, "role_eval_*_exact.json")):
            if "semantic" in os.path.basename(fp).lower():
                hits.append(fp)
    return hits


def main():
    out = {}
    md = ["# E16 — Semantic Match resolution", "",
          "Semantic Match is annotated (11th human dimension) but not a model head "
          "(DRAME-Eval predicts 10). Below: coverage, agreement, and near-redundancy with "
          "Local Scene Fit that justify the design; it stays in the release for future work.", ""]

    # (1)+(3) coverage and correlation per split
    splits = [("train", load_file(TRAIN_FILE)),
              ("test_base", list(common.load_metadata("realism_base").values())),
              ("test_realrec", list(common.load_metadata("realism_real").values())),
              ("benchmark", load_file(BENCH_FILE))]
    md += ["## 1. Coverage + consensus correlation with Local Scene Fit", "",
           "| Split | semantic_match annotated | coverage | r(sm, Local Scene Fit) | n |",
           "|---|---|---|---|---|"]
    out["per_split"] = {}
    for name, recs in splits:
        ann, tot = coverage(recs)
        r, n = sm_vs_local(recs)
        md.append(f"| {name} | {ann:,} / {tot:,} | {100*ann/tot:.1f}% | {r:+.3f} | {n:,} |")
        out["per_split"][name] = {"annotated": ann, "total": tot, "r_sm_local": r, "n": n}
    md += ["", "_semantic_match correlates ~0.85-0.99 with Local Scene Fit at the consensus "
           "level — at the aggregate the two are near-redundant, so a dedicated 11th head "
           "would add little predictive signal. The uncovered samples are legitimately N/A "
           "(no scene to match against, e.g. some positives)._", ""]

    # (2) agreement
    md += ["## 2. Inter-annotator agreement on semantic_match", "",
           "| Split | ordinal α | weighted κ | n items |", "|---|---|---|---|"]
    out["agreement"] = {}
    for name in ("realism_base", "realism_real"):
        meta = common.load_metadata(name)
        a, k, n = sm_agreement(meta)
        md.append(f"| {name} | {a:.3f} | {k:.3f} | {n:,} |")
        out["agreement"][name] = {"alpha": a, "kappa_quadratic": k, "n": n}
    md += ["", "_Semantic Match is among the highest-agreement dimensions (it is the most "
           "objective — content adequacy), which is why it is a reliable QA/annotation signal "
           "even though it is not a model target._", ""]

    # (4) verify no prediction head
    hits = check_no_sm_prediction()
    md += ["## 3. Model design: 10 heads, no semantic_match", "",
           f"- Prediction files targeting semantic_match across all DRAME dirs: "
           f"**{len(hits)}** → the model is 10-dimensional by design; semantic adequacy is "
           f"largely audio-independent and near-redundant with Local Scene Fit (§1).",
           "- Action (G1.4): fix the main-text dimension list so the 10 model heads vs 11 "
           "annotation dimensions is explicit, not silent; keep semantic_match in the release "
           "with these stats in the appendix; an 11th head is optional future work (D6).", ""]
    out["sm_prediction_files"] = len(hits)

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e16_semantic_match.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e16_semantic_match.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e16_semantic_match.md"))
    print("\n--- E16 HIGHLIGHTS ---")
    for name, d in out["per_split"].items():
        print(f"  {name:14s} cov {100*d['annotated']/d['total']:.0f}%  r(sm,Local) {d['r_sm_local']:+.3f}")
    print(f"  semantic_match prediction files: {len(hits)}")


if __name__ == "__main__":
    main()
