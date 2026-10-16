"""
E15 — leave-one-system-out analysis (reviewer gEEq; archetype system-shared concern).
Both models trained with the identical recipe (LoRA, 2000 steps) differing ONLY in whether
Doubao was in training:
  - control  (job-n2jll0k87z): all 8 systems in train  -> Doubao is SEEN
  - loso     (job-35hrrdnl40): Doubao held out of train -> Doubao is UNSEEN
Both evaluated on the Doubao test subset (200). The control−loso gap on Doubao is the
generalization drop attributable to never having seen the system.

Needs results_arche/loso_control_ckpt2000_doubao/ + loso_noDoubao_ckpt2000_doubao/. Offline.
Writes analysis/out/e15_loso.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common

CTRL = os.path.join(common.EVAL_DIR, "results_arche/loso_control_ckpt2000_doubao")
LOSO = os.path.join(common.EVAL_DIR, "results_arche/loso_noDoubao_ckpt2000_doubao")
DIMS = ["audio_quality", "human_likeness", "appropriateness"]


def load_arche(d, dim):
    fp = os.path.join(d, f"arche_eval_{dim}_exact.json")
    if not os.path.exists(fp):
        return None
    return {common._norm_id(r["id"]): float(r["probability_score"]) for r in json.load(open(fp))}


def corr_on(preds_dir, meta, doubao_ids):
    res = {}
    for dim in DIMS:
        preds = load_arche(preds_dir, dim)
        if preds is None:
            res[dim] = float("nan"); continue
        P, G = [], []
        for rid in doubao_ids:
            if rid in preds and rid in meta:
                v = meta[rid].get("annotations", {}).get(dim)
                if v is not None:
                    P.append(preds[rid]); G.append(float(v))
        res[dim] = common.pearson(P, G) if len(P) > 2 else float("nan")
    return res


def main():
    if not os.path.exists(os.path.join(LOSO, "arche_eval_appropriateness_exact.json")):
        print("E15 LOSO eval preds not present yet — waiting on job-35hrrdnl40 / job-n2jll0k87z."); return
    meta = common.load_metadata("archetype")
    doubao_ids = {rid for rid, r in meta.items() if r.get("sfnd", {}).get("system") == "Doubao"}
    ctrl = corr_on(CTRL, meta, doubao_ids)
    loso = corr_on(LOSO, meta, doubao_ids)
    md = ["# E15 — leave-one-system-out (Doubao) retrain", "",
          f"Identical recipe (LoRA, 2000 steps), Doubao {len(doubao_ids)} test samples. "
          f"control = Doubao seen in training; loso = Doubao held out. Pearson vs human.", "",
          "| Dim | control (Doubao seen) | loso (Doubao UNSEEN) | Δ (loso−control) |", "|---|---|---|---|"]
    out = {"control": ctrl, "loso": loso, "n_doubao": len(doubao_ids)}
    for dim in DIMS:
        d = loso[dim] - ctrl[dim]
        md.append(f"| {common.ARCHE_LABELS[dim]} | {ctrl[dim]:+.3f} | {loso[dim]:+.3f} | {d:+.3f} |")
    ca = float(np.nanmean([ctrl[d] for d in DIMS])); la = float(np.nanmean([loso[d] for d in DIMS]))
    md.append(f"| **avg** | **{ca:+.3f}** | **{la:+.3f}** | **{la-ca:+.3f}** |")
    out["control_avg"] = ca; out["loso_avg"] = la; out["gap"] = la - ca
    retain = 100 * la / ca if ca else float("nan")
    md += ["", f"_On the held-out Doubao system, the model that NEVER saw Doubao in training "
           f"scores **{la:+.3f}** vs **{ca:+.3f}** for the model that did — retaining "
           f"~{retain:.0f}% of the seen-system correlation. "
           f"{'The judge generalizes to an unseen synthesis system (the system-shared concern is largely defused).' if la > 0.4 and (la >= 0.7*ca or ca<=0) else 'Generalization to the unseen system is substantially weaker — concede the system-shared limitation and narrow the claim.'}_", ""]
    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e15_loso.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e15_loso.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e15_loso.md"))
    print(f"  Doubao: control {ca:+.3f}  loso(unseen) {la:+.3f}  gap {la-ca:+.3f}")


if __name__ == "__main__":
    main()
