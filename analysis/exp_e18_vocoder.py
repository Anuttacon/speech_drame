"""
E18 — mel/vocoder fidelity: does BigVGAN reconstruction change DRAME-Eval's scores?
Compares DRAME on ORIGINAL audio vs BigVGAN-RECONSTRUCTED audio for the same ~200 test_base
samples (job-4mk14k8y7g). If scores are preserved (low drift, high rank corr), the released
mel-based reconstructions are faithful for scoring — supporting the release design.

Needs results/e18_recon_drame/. Runs offline.
Writes analysis/out/e18_vocoder.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from exp_e1_bootstrap import load_dir_preds, realism_human, build_cells

ORIG = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"
RECON = "results/e18_recon_drame"


def main():
    if not os.path.exists(os.path.join(common.EVAL_DIR, RECON, "role_eval_pitch_dynamics_exact.json")):
        print("E18 recon preds not present yet — waiting on job-4mk14k8y7g."); return
    orig = load_dir_preds(ORIG, common.REALISM_DIMS)
    recon = load_dir_preds(RECON, common.REALISM_DIMS)
    meta = common.load_metadata("realism_base")
    ids = [i for i in recon if i in orig]  # the ~200 reconstructed
    md = ["# E18 — original vs BigVGAN-reconstructed DRAME scores", "",
          f"Same {len(ids)} test_base samples scored on original audio vs BigVGAN "
          f"reconstruction from the released mel. Score drift + rank preservation + effect on "
          f"human-correlation.", "",
          "| Dim | mean|Δ score| | Pearson(orig,recon) | r_human orig | r_human recon |",
          "|---|---|---|---|---|"]
    out = {"n": len(ids), "dims": {}}
    drifts, corrs = [], []
    for d in common.REALISM_DIMS:
        O, R, G = [], [], []
        for i in ids:
            if d in orig[i] and d in recon[i]:
                O.append(orig[i][d]); R.append(recon[i][d])
                g = common.human_mean(meta[i], d, "mask") if i in meta else None
                G.append(g)
        O, R = np.array(O), np.array(R)
        drift = float(np.mean(np.abs(O - R)))
        corr = common.pearson(O, R)
        # human corr on the subset for orig vs recon
        mask = [g is not None for g in G]
        Gm = [g for g in G if g is not None]
        rho_o = common.pearson(O[mask], Gm) if sum(mask) > 2 else float("nan")
        rho_r = common.pearson(R[mask], Gm) if sum(mask) > 2 else float("nan")
        out["dims"][d] = {"drift": drift, "corr_orig_recon": corr, "r_human_orig": rho_o, "r_human_recon": rho_r}
        drifts.append(drift); corrs.append(corr)
        md.append(f"| {common.REALISM_LABELS[d]} | {drift:.3f} | {corr:+.3f} | {rho_o:+.3f} | {rho_r:+.3f} |")
    md.append(f"| **mean** | **{np.mean(drifts):.3f}** | **{np.nanmean(corrs):+.3f}** | — | — |")
    md += ["", f"_Mean absolute score drift {np.mean(drifts):.3f} on the 1-5 scale; "
           f"orig-vs-recon per-sample correlation {np.nanmean(corrs):+.3f}. "
           f"{'Reconstruction preserves DRAME scores well — released mel reconstructions are faithful for scoring.' if np.mean(drifts)<0.3 and np.nanmean(corrs)>0.8 else 'Reconstruction shifts scores; flag in release notes.'}_", ""]
    out["mean_drift"] = float(np.mean(drifts)); out["mean_corr"] = float(np.nanmean(corrs))
    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e18_vocoder.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e18_vocoder.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e18_vocoder.md"))
    print(f"  n={len(ids)} mean|Δ|={np.mean(drifts):.3f} corr(orig,recon)={np.nanmean(corrs):+.3f}")


if __name__ == "__main__":
    main()
