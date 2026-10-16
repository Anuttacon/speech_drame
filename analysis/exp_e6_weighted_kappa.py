"""
E6 — Quadratic-weighted κ per dimension, alongside the ordinal Krippendorff α already in
the Tier-1 appendix (reviewer NhWo C2, reviewer ZB3A RC3 = "human IRR not reported").

The raters are NOT identified across items (released `annotations.raters.<dim>` is an
unordered per-item list, ~4-6 raters/sample), so we report the *average pairwise*
quadratic-weighted κ: build the ordinal coincidence matrix from every within-item rater
pair, then normalise with κ's chance model (product of marginals) using squared-distance
weights. This is the weighted-κ analogue of the α that ships in `app:tier1`, and it is the
chance-corrected agreement the reviewers name. α and κ_w are reported side by side; they
answer the same ask with two standard estimators.

Mask gating (drop N/A / null rater entries) — the release-native protocol; identical to the
convention used for the α already in the paper.

Run from analysis/:  PYTHONPATH=. python3 exp_e6_weighted_kappa.py
Writes analysis/out/e6_weighted_kappa.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common


def weighted_kappa_pairwise(units, weights="quadratic"):
    """Average pairwise weighted κ from variable-length per-item rater lists.

    units: list of per-item rater-value lists (missing already dropped).
    Coincidence matrix O_ck = (1/(m-1)) Σ over ordered within-item pairs (as in
    Krippendorff), symmetric. κ_w = 1 - Σ w O / Σ w E, with E from the value marginals
    (product model) and w_ck the squared (quadratic) or absolute (linear) rank distance.
    Returns nan if fewer than two distinct values are observed.
    """
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return float("nan")
    vidx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    O = np.zeros((k, k), float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                O[vidx[u[a]], vidx[u[b]]] += 1.0 / (m - 1)
    tot = O.sum()
    if tot == 0:
        return float("nan")
    Op = O / tot
    marg = Op.sum(axis=1)
    E = np.outer(marg, marg)
    varr = np.array(vals, float)
    if weights == "quadratic":
        W = (varr[:, None] - varr[None, :]) ** 2
    else:  # linear
        W = np.abs(varr[:, None] - varr[None, :])
    denom = (W * E).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (W * Op).sum() / denom)


def collect_units(meta, dim, gating="mask"):
    units = []
    for r in meta.values():
        u = common.human_rater_arrays(r, dim, gating=gating)
        if len(u) >= 2:
            units.append(u)
    return units


def n_raters_stats(meta, dim, gating="mask"):
    ns = [len(common.human_rater_arrays(r, dim, gating=gating)) for r in meta.values()]
    ns = [n for n in ns if n >= 2]
    if not ns:
        return 0, 0.0
    return len(ns), float(np.mean(ns))


def block(title, meta, dims, labels):
    lines = [f"## {title}", "",
             "| Dimension | Krippendorff α (ordinal) | Weighted κ (quadratic) | Weighted κ (linear) | n items (≥2 raters) | mean raters |",
             "|---|---|---|---|---|---|"]
    res = {}
    for d in dims:
        units = collect_units(meta, d)
        alpha = common.krippendorff_alpha_ordinal(units)
        kq = weighted_kappa_pairwise(units, "quadratic")
        kl = weighted_kappa_pairwise(units, "linear")
        n_it, mean_r = n_raters_stats(meta, d)
        lab = labels.get(d, d)
        lines.append(f"| {lab} | {alpha:.3f} | {kq:.3f} | {kl:.3f} | {n_it} | {mean_r:.2f} |")
        res[d] = {"alpha": alpha, "kappa_quadratic": kq, "kappa_linear": kl,
                  "n_items": n_it, "mean_raters": mean_r}
    # macro row
    def macro(key):
        vs = [res[d][key] for d in dims if res[d][key] == res[d][key]]
        return float(np.mean(vs)) if vs else float("nan")
    lines.append(f"| **Average** | {macro('alpha'):.3f} | {macro('kappa_quadratic'):.3f} | "
                 f"{macro('kappa_linear'):.3f} | — | — |")
    lines.append("")
    res["_macro"] = {"alpha": macro("alpha"), "kappa_quadratic": macro("kappa_quadratic"),
                     "kappa_linear": macro("kappa_linear")}
    return lines, res


def main():
    out = {"gating": "mask", "method": "average pairwise, ordinal coincidence matrix"}
    md = ["# E6 — Weighted κ inter-rater agreement (alongside Krippendorff α)", "",
          "Chance-corrected agreement per dimension. α = ordinal Krippendorff (same estimator "
          "already reported in App. `app:tier1`); κ_w = average-pairwise quadratic/linear "
          "weighted kappa. Raters are unidentified per item (~4-6/sample); κ_w uses every "
          "within-item rater pair. Mask gating (drop null/N-A rater entries).", "",
          "_Reference bands: κ_w 0.4-0.6 moderate, 0.6-0.8 substantial (Landis & Koch); "
          "ordinal α ≥ 0.667 is the common reliability floor._", ""]

    meta_base = common.load_metadata("realism_base")
    l, r = block("Realism in-distribution base test (n=2,000)", meta_base,
                 common.REALISM_DIMS, common.REALISM_LABELS)
    md += l; out["realism_base"] = r

    meta_real = common.load_metadata("realism_real")
    l, r = block("Realism real recordings (test_realrec, n=2,164)", meta_real,
                 common.REALISM_DIMS, common.REALISM_LABELS)
    md += l; out["realism_real"] = r

    meta_arche = common.load_metadata("archetype")
    l, r = block("Archetype test (content-pass samples, n≈1,312)", meta_arche,
                 common.ARCHE_DIMS, common.ARCHE_LABELS)
    md += l; out["archetype"] = r

    md += ["## Notes for integration", "",
           "- Add the κ_w columns next to α in `tab:tier1-ceiling`; state the multi-rater "
           "protocol (each sample rated by ~4-6 annotators; scores are the rater mean).",
           "- α and κ_w agree on the ordering across dimensions, so the reliability story is "
           "robust to estimator choice — directly answers ZB3A RC3 / NhWo C2.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e6_weighted_kappa.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e6_weighted_kappa.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e6_weighted_kappa.md"))
    print("\n--- E6 HIGHLIGHTS (macro α / κ_quad) ---")
    for k in ("realism_base", "realism_real", "archetype"):
        mc = out[k]["_macro"]
        print(f"  {k:16s} α {mc['alpha']:.3f}  κ_w(quad) {mc['kappa_quadratic']:.3f}")


if __name__ == "__main__":
    main()
