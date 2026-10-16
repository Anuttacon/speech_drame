"""
Experiment 1 — Human-agreement ceiling.

For every test split and every scored dimension we estimate how well a *single human*
agrees with the consensus of the remaining raters, which provides a descriptive human reference for the
*model*-vs-consensus Pearson reported in the paper's Tables 2-4. We also report ordinal
Krippendorff's alpha (the chance-corrected IAA the paper notes is missing), and express the
final DRAME-Eval correlation as a fraction of the human ceiling.

Two reliability estimands per dimension:
  * single-vs-rest Pearson : pool (held-out rater, mean-of-rest) pairs over all raters/samples.
                             Directly comparable in shape to model-vs-mean Pearson.
  * Krippendorff ordinal alpha : computed on the per-sample rater arrays.

Emotion Intensity/Transition are gated (N/A floored to 1). We report those two under both
'mask' (drop N/A) and 'impute' (N/A->1) conventions; all other dims use 'mask'.

Outputs: analysis/out/exp1_human_ceiling.json  and  analysis/out/exp1.md
"""
import json
import os

import numpy as np

import common as C


SPLITS = [
    ("archetype", C.ARCHE_DIMS, C.ARCHE_LABELS),
    ("realism_base", C.REALISM_DIMS, C.REALISM_LABELS),
    ("realism_real_v2", C.REALISM_DIMS, C.REALISM_LABELS),
]
LANGS = [("all", None), ("en", "en"), ("zh", "zh")]


def _unit_pairs(records, dim, gating):
    """Per-unit (held_out, mean_of_rest) pair arrays, for units with >=2 valid raters.

    Returns (units, H, R, owner) where H/R are the flat pooled pairs and owner[i] is the
    unit index that produced pair i (used for fast sample-level bootstrap).
    """
    units, H, R, owner = [], [], [], []
    for rec in records:
        arr = C.human_rater_arrays(rec, dim, gating=gating)
        if len(arr) < 2:
            continue
        u = len(units)
        units.append(arr)
        a = np.array(arr, float)
        for i in range(len(a)):
            H.append(a[i])
            R.append((a.sum() - a[i]) / (len(a) - 1))
            owner.append(u)
    return units, np.array(H), np.array(R), np.array(owner, int)


def ceiling_point_ci(records, dim, gating, seed=0, n=1000):
    """Single-vs-rest Pearson point estimate + sample-level bootstrap CI (vectorized)."""
    units, H, R, owner = _unit_pairs(records, dim, gating)
    n_units = len(units)
    if n_units < 2 or len(H) < 2:
        return float("nan"), (float("nan"), float("nan")), n_units
    point = C.pearson(H, R)
    # group pair-indices by owning unit once
    by_unit = [np.where(owner == u)[0] for u in range(n_units)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        sel = rng.integers(0, n_units, n_units)
        idx = np.concatenate([by_unit[u] for u in sel])
        stats[b] = C.pearson(H[idx], R[idx])
    stats = stats[np.isfinite(stats)]
    lo, hi = (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))) if len(stats) else (np.nan, np.nan)
    return point, (lo, hi), n_units


def run():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    results = {}
    md = ["## Experiment 1 — Human-agreement ceiling\n"]
    md.append(
        "_Single-vs-rest Pearson_ = a held-out rater vs the mean of the remaining raters "
        "(pooled over raters and samples); this is a descriptive human reference for the "
        "model-vs-consensus Pearson in Tables 2-4. _K-alpha_ = ordinal Krippendorff's alpha. "
        "_DRAME-Eval_ = final Instruct-LoRA Pearson vs the full-consensus mean. "
        "_%ceil_ = DRAME-Eval / single-vs-rest. Brackets are 95% bootstrap CIs "
        "(1000 resamples, sample-level). Human N/A values are masked. The authentic "
        "evaluation is restricted to clean test_real_v2 IDs; full-pool historical human "
        "reference values can differ.\n")

    for split, dims, labels in SPLITS:
        meta = C.load_metadata(split)
        if split == "realism_real_v2":
            from realrec_v2_clean import load_v2_ids
            keep = set(load_v2_ids())
            meta = {rid: rec for rid, rec in meta.items() if rid in keep}
        drame = C.load_drame_preds(split)
        results[split] = {}
        md.append(f"\n### {split}  (n={len(meta)})\n")
        md.append("| Lang | Dimension | n | single-vs-rest | K-alpha | DRAME-Eval | %ceil |")
        md.append("|---|---|---|---|---|---|---|")

        langs_present = {r.get("language") for r in meta.values()}
        single_lang = len([x for x in langs_present if x]) <= 1
        for lang_name, lang in LANGS:
            if lang is not None and single_lang:
                continue  # avoid duplicate rows when the split is one language only
            recs = [r for r in meta.values() if (lang is None or r.get("language") == lang)]
            if not recs:
                continue
            results[split][lang_name] = {}
            for dim in dims:
                for gating in ["mask"]:  # release arrays carry no N/A -> mask == impute
                    ceil, (lo, hi), n_units = ceiling_point_ci(recs, dim, gating)
                    units = [u for u in (C.human_rater_arrays(r, dim, gating=gating) for r in recs) if len(u) >= 2]
                    kalpha = C.krippendorff_alpha_ordinal(units)
                    # DRAME-Eval vs full consensus mean (same gating)
                    P, G = [], []
                    for rid, pr in drame.items():
                        if dim not in pr:
                            continue
                        rec = meta.get(rid)
                        if rec is None or (lang is not None and rec.get("language") != lang):
                            continue
                        g = C.human_mean(rec, dim, gating=gating)
                        if g is None:
                            continue
                        P.append(pr[dim])
                        G.append(g)
                    dr = C.pearson(P, G)
                    pct = (dr / ceil * 100) if (ceil == ceil and ceil > 0) else float("nan")
                    tag = labels.get(dim, dim)
                    results[split][lang_name][tag] = {
                        "single_vs_rest": ceil, "ci": [lo, hi], "kalpha": kalpha,
                        "drame_eval": dr, "pct_ceiling": pct, "n_units": n_units, "n_model": len(P),
                    }
                    md.append(
                        f"| {lang_name} | {tag} | {n_units} | {C.fmt_ci(ceil, lo, hi)} | "
                        f"{kalpha:.3f} | {dr:.3f} | {pct:.0f}% |")

    with open(os.path.join(C.OUT_DIR, "exp1_human_ceiling.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(C.OUT_DIR, "exp1.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n[written] analysis/out/exp1_human_ceiling.json, exp1.md")


if __name__ == "__main__":
    run()
