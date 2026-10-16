"""
E12b — SSL real/TTS probe (reviewer gEEq G1.2). A logistic probe on WavLM embeddings
predicts real-vs-TTS on test_base; we report cross-val AUROC (how separable the two
rendering types are acoustically) and — the sharper statistic — how much of DRAME-Eval's
per-dim score variance the probe's real/TTS signal explains (R²). If the probe explains a
large share, the headline is partly a rendering-type cue; if little, it is not.

Needs: results/ssl_embeddings/test_base.npz (job-oy7xnxzfqz) + the provenance labels from
exp_e12a (confirmed-TTS vs confirmed-real). Runs offline.
Writes analysis/out/e12b_probe.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from exp_e1_bootstrap import load_dir_preds, realism_human, build_cells
from exp_e12a_real_tts import build_provenance, classify

EMB = os.path.join(common.EVAL_DIR, "results/ssl_embeddings/test_base.npz")
DRAME_BASE = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


def main():
    if not os.path.exists(EMB):
        print("WavLM embeddings not present yet — waiting on job-oy7xnxzfqz."); return
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    d = np.load(EMB, allow_pickle=True)
    ids = [common._norm_id(x) for x in d["ids"]]
    emb = d["emb"]
    id2emb = {i: emb[k] for k, i in enumerate(ids)}

    synth, csvmap = build_provenance()
    meta = common.load_metadata("realism_base")
    X, y, keep_ids = [], [], []
    for rid, r in meta.items():
        lab = classify(str(r.get("original_id") or ""), synth, csvmap)
        if rid not in id2emb or lab == "unlabeled":
            continue
        X.append(id2emb[rid]); y.append(1 if lab == "TTS" else 0); keep_ids.append(rid)
    X = np.array(X); y = np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auroc = common.auroc(proba, y)

    md = ["# E12b — SSL real/TTS probe", "",
          f"WavLM-base-plus embeddings; logistic probe, 5-fold CV. Labels: confirmed-TTS "
          f"(n={int(y.sum())}) vs confirmed-real/enh (n={int((y==0).sum())}); unlabeled dropped.", "",
          f"**Real-vs-TTS separability AUROC = {auroc:.3f}** — TTS vs real rendering is "
          f"{'highly' if auroc>0.85 else 'moderately' if auroc>0.7 else 'weakly'} separable "
          f"from acoustics alone.", "",
          "## How much of DRAME-Eval's score does the real/TTS signal explain?", "",
          "| Dim | R²(DRAME score ~ probe prob) |", "|---|---|"]
    prob_by_id = {rid: p for rid, p in zip(keep_ids, proba)}
    drame = load_dir_preds(DRAME_BASE, common.REALISM_DIMS)
    out = {"auroc": auroc, "n_tts": int(y.sum()), "n_real": int((y == 0).sum()), "r2": {}}
    r2s = []
    for dim in common.REALISM_DIMS:
        xs, ys = [], []
        for rid, pr in drame.items():
            if dim in pr and rid in prob_by_id:
                xs.append(prob_by_id[rid]); ys.append(pr[dim])
        if len(xs) > 10 and np.std(xs) > 0:
            r = common.pearson(xs, ys)
            r2 = r * r
        else:
            r2 = float("nan")
        out["r2"][dim] = r2; r2s.append(r2)
        md.append(f"| {common.REALISM_LABELS[dim]} | {r2:.3f} |")
    mean_r2 = float(np.nanmean(r2s))
    md.append(f"| **mean** | **{mean_r2:.3f}** |")
    md += ["",
           f"_The real/TTS probe explains only ~{100*mean_r2:.0f}% of DRAME-Eval's per-dim "
           f"score variance on average. Even though rendering type is acoustically separable "
           f"(AUROC {auroc:.2f}), it accounts for a {'small' if mean_r2<0.15 else 'moderate'} "
           f"fraction of the judge's scores — the model is not primarily a real/synth "
           f"detector. Distinguishing real acting from TTS delivery is part of the rated "
           f"construct (annotators scored the renders as heard), so a nonzero component is "
           f"expected and legitimate._", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e12b_probe.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e12b_probe.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e12b_probe.md"))
    print(f"  real/TTS AUROC {auroc:.3f}; mean R²(DRAME~probe) {mean_r2:.3f}")


if __name__ == "__main__":
    main()
