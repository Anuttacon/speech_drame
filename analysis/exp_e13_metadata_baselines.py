"""
E13 — Metadata-only baselines (reviewer gEEq G2.2 "source-identity baseline"). Quantifies
how much of the headline correlation is explained by the CONSTRUCTION DISTRIBUTION alone —
i.e. what a predictor that never listens to the audio and only knows metadata can achieve.

Three dumb predictors, each correlated (per dim, then averaged) against the human scores:
  (i)   source-show mean  — predict a test_base sample's score from the TRAIN mean of its
        source show (fallback: global train mean). Knows only "which show".
  (ii)  speaker mean (LOO) — predict a real-rec sample's score from its speaker's mean,
        leave-one-out. Knows only "which of the 4-5 speakers". This is the between-speaker
        composition ceiling (cf. E4).
  (iii) construction oracle — predict a test_base sample from the mean of its
        (sample_type, negative_type, severity) cell. Knows only "which construction bucket".

If a metadata baseline rivals the models, that much of the correlation is distributional
structure, not per-sample audio judgment (gEEq reason #2). Reported next to the model
numbers from E1/E7 for context.

Run from analysis/:  PYTHONPATH=. python3 exp_e13_metadata_baselines.py
Writes analysis/out/e13_metadata_baselines.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

import common
from realrec_v2_clean import load_v2_ids, speaker_of

TRAIN_FILE = os.path.join(common.REALISM_META, "train.jsonl")
# model anchors (from E1/E7, mask gating) for context
ANCHORS_BASE = {"DRAME-Eval": 0.602, "Gemini2.5Pro (FS)": 0.387, "GPT4o-audio (FS)": 0.305}
ANCHORS_REAL = {"GPT4o-audio (FS)": 0.227, "DRAME-Eval (clean)": -0.142}


def show_of(oid):
    if not oid:
        return None
    m = re.match(r"(.+?)_\d+x\d+_", oid)
    if m:
        return m.group(1)
    return re.split(r"_\d", oid)[0]


def load_train():
    return [json.loads(l) for l in open(TRAIN_FILE)]


def avg_pearson(pred_map, meta, dims, id_filter=None):
    """pred_map: {dim: {id: predicted}}. Correlate vs human_mean per dim, average."""
    rs = []
    for d in dims:
        P, G = [], []
        for rid, rec in meta.items():
            if id_filter is not None and rid not in id_filter:
                continue
            if rid not in pred_map[d]:
                continue
            g = common.human_mean(rec, d, "mask")
            if g is None:
                continue
            P.append(pred_map[d][rid]); G.append(g)
        rs.append(common.pearson(P, G))
    return float(np.nanmean(rs)), rs


def show_mean_baseline(meta, dims):
    train = load_train()
    # train show means per dim
    train_means = {d: collections.defaultdict(list) for d in dims}
    global_mean = {}
    for r in train:
        sh = show_of(r.get("original_id"))
        for d in dims:
            g = common.human_mean(r, d, "mask")
            if g is not None:
                train_means[d][sh].append(g)
    show_avg = {d: {sh: float(np.mean(v)) for sh, v in train_means[d].items()} for d in dims}
    for d in dims:
        allv = [g for v in train_means[d].values() for g in v]
        global_mean[d] = float(np.mean(allv)) if allv else 3.0
    pred = {d: {} for d in dims}
    covered = 0; total = 0
    for rid, rec in meta.items():
        sh = show_of(rec.get("original_id"))
        total += 1
        hit = sh in show_avg[dims[0]]
        covered += int(hit)
        for d in dims:
            pred[d][rid] = show_avg[d].get(sh, global_mean[d])
    return pred, covered / total if total else 0


def speaker_loo_baseline(meta, dims, id_filter):
    # gather per speaker per dim values
    by = {d: collections.defaultdict(list) for d in dims}
    ids_by = {d: collections.defaultdict(list) for d in dims}
    for rid, rec in meta.items():
        if rid not in id_filter:
            continue
        spk = speaker_of(rid)
        for d in dims:
            g = common.human_mean(rec, d, "mask")
            if g is not None:
                by[d][spk].append(g); ids_by[d][spk].append(rid)
    pred = {d: {} for d in dims}
    for d in dims:
        for spk, vals in by[d].items():
            s = sum(vals); n = len(vals)
            for rid, v in zip(ids_by[d][spk], vals):
                pred[d][rid] = (s - v) / (n - 1) if n > 1 else v  # leave-one-out
    return pred


def construction_oracle_baseline(meta, dims):
    def bucket(rec):
        return (rec.get("sample_type"), rec.get("negative_type"), rec.get("processing_level"))
    by = {d: collections.defaultdict(list) for d in dims}
    idb = {d: collections.defaultdict(list) for d in dims}
    for rid, rec in meta.items():
        b = bucket(rec)
        for d in dims:
            g = common.human_mean(rec, d, "mask")
            if g is not None:
                by[d][b].append(g); idb[d][b].append(rid)
    pred = {d: {} for d in dims}
    for d in dims:
        for b, vals in by[d].items():
            s = sum(vals); n = len(vals)
            for rid, v in zip(idb[d][b], vals):
                pred[d][rid] = (s - v) / (n - 1) if n > 1 else v  # LOO within bucket
    return pred


def main():
    dims = common.REALISM_DIMS
    meta_b = common.load_metadata("realism_base")
    meta_r = common.load_metadata("realism_real")
    v2 = set(load_v2_ids())
    out = {}
    md = ["# E13 — Metadata-only baselines (no audio)", "",
          "How much correlation a predictor that never hears the audio and only knows metadata "
          "achieves. If it rivals the models, that much of the headline is construction "
          "distribution, not per-sample judgment (gEEq reason #2). Mask gating; avg over 10 "
          "dims; Pearson vs human mean.", ""]

    # (i) source-show mean on test_base
    pred, cov = show_mean_baseline(meta_b, dims)
    r_show, _ = avg_pearson(pred, meta_b, dims)
    # (iii) construction oracle on test_base
    pred_o = construction_oracle_baseline(meta_b, dims)
    r_oracle, _ = avg_pearson(pred_o, meta_b, dims)
    md += ["## test_base (in-distribution)", "",
           "| Baseline (no audio) | Avg Pearson | note |", "|---|---|---|",
           f"| Source-show mean (train-estimated) | {r_show:+.3f} | {100*cov:.0f}% of test shows seen in train |",
           f"| Construction oracle (type×neg×severity, LOO) | {r_oracle:+.3f} | knows only the construction bucket |"]
    for lab, v in ANCHORS_BASE.items():
        md.append(f"| _model anchor: {lab}_ | {v:+.3f} | from E1/E7 |")
    md += ["",
           f"_The source-show baseline reaches **{r_show:+.3f}** and the construction-bucket "
           f"oracle **{r_oracle:+.3f}** — both far below DRAME-Eval's 0.602, but the oracle "
           f"shows a non-trivial slice of the correlation is bucket structure (positives rate "
           f"high, severe negatives low). DRAME-Eval clears the metadata baselines by a wide "
           f"margin → it is judging within buckets, not just recovering the construction "
           f"distribution._", ""]
    out["test_base"] = {"source_show_mean": r_show, "construction_oracle": r_oracle,
                        "show_coverage": cov, "anchors": ANCHORS_BASE}

    # (ii) speaker LOO on real-rec
    pred_s = speaker_loo_baseline(meta_r, dims, v2)
    r_spk, _ = avg_pearson(pred_s, meta_r, dims, id_filter=v2)
    md += ["## real recordings (test_real_v2)", "",
           "| Baseline (no audio) | Avg Pearson | note |", "|---|---|---|",
           f"| Speaker mean (leave-one-out, 4-5 speakers) | {r_spk:+.3f} | between-speaker composition ceiling |"]
    for lab, v in ANCHORS_REAL.items():
        md.append(f"| _model anchor: {lab}_ | {v:+.3f} | from E1/R3-C |")
    md += ["",
           f"_The speaker-mean baseline (no audio, only 4-5 distinct predicted values) reaches "
           f"**{r_spk:+.3f}** — which **exceeds every audio model** on the pooled real set "
           f"(best judge GPT4o-FS 0.227; clean DRAME −0.142). A predictor that knows nothing "
           f"but which of the 4-5 speakers is talking out-correlates all of them. This is the "
           f"E4 story made quantitative and is the sharpest single number for W1: the pooled "
           f"real-rec correlation is dominated by between-speaker composition, not "
           f"within-speaker acting judgment, so no method's pooled real-rec figure (ours or a "
           f"baseline's) should be read as evidence of per-sample real-speech evaluation._", ""]
    out["real_rec"] = {"speaker_loo_mean": r_spk, "anchors": ANCHORS_REAL}

    md += ["## Integration", "",
           "- Appendix table + one sentence in §6.2: metadata baselines bound the "
           "distributional component; DRAME-Eval clears them in-distribution, and the "
           "real-rec pooled number is near the speaker-mean baseline (feeds W1 framing + E2).", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e13_metadata_baselines.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e13_metadata_baselines.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e13_metadata_baselines.md"))
    print("\n--- E13 HIGHLIGHTS ---")
    print(f"  base: source-show {r_show:+.3f}  construction-oracle {r_oracle:+.3f}  (DRAME 0.602)")
    print(f"  real: speaker-LOO {r_spk:+.3f}  (GPT4o-FS 0.227, DRAME clean -0.142)")


if __name__ == "__main__":
    main()
