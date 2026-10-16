"""
Clean real-recording evaluation of the final DRAME-Eval Instruct-LoRA on the canonical
held-out test_real_v2 (1,084 = qyf 271 / zrr 271 / zyx 270 / yy 270 / pro-mufei 2).

Motivation (handoffs/rebuttal_r1.md §3): the paper's published real-rec DRAME-Eval 0.247
was computed on the OLD v1 test list, 540/1,084 of whose samples are in the v2 model's own
training set -> train-contaminated. This script recomputes the number on the fully held-out
test_real_v2 partition, using predictions produced by cluster job job-pqa5dxofkv.

It reports:
  (1) HEADLINE  - pooled per-dim + overall Pearson on all 1,084 held-out samples.
  (2) OVERLAP   - cross-check on the 544 ids shared with the contaminated dir
                  (zrr+qyf+2). Same model, same audio -> predictions must be ~identical
                  and reproduce the interim held-out r (~0.049). A large divergence here
                  means the two runs are not the same checkpoint.
  (3) NEW-ONLY  - pooled r on the 540 previously-unscored samples (yy+zyx): the first
                  clean number for those held-out speakers.
  (4) PER-SPEAKER - within-speaker r decomposition (E2): pooled r is inflated by across-
                  speaker mean differences; report mean-within-speaker vs pooled.
  (5) Both emotion-gating conventions (impute = paper, mask = conservative) for the overall.

Run from analysis/:  python3 realrec_v2_clean.py
The clean split key 'realism_real_v2' is registered in common.py (DRAME_DIRS/META_FILE).
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common

CLEAN_SPLIT = "realism_real_v2"
CONTAM_SPLIT = "realism_real"          # train-contaminated v1-list run (baseline for cross-check)
HUMAN_CEILING_REF = 0.561              # Pooled single-rater reference; within-speaker is ~0.34
V2_DATA = os.path.join(common.EVAL_DIR, "data/test_real_v2/data.jsonl")


def speaker_of(rid: str) -> str:
    """role-eval_v6_<spk>_<date>_...  ->  <spk>   (robust to the vN version token)."""
    p = rid.replace("role-eval_", "").split("_")
    return p[1] if p and p[0].startswith("v") and p[0][1:].isdigit() else p[0]


def load_v2_ids():
    ids = []
    with open(V2_DATA) as f:
        for line in f:
            ids.append(common._norm_id(json.loads(line)["id"]))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Clean test manifest is empty or contains duplicate IDs")
    train_path = os.path.join(common.EVAL_DIR, "data/train_all_real_v2/data.jsonl")
    with open(train_path) as f:
        train = {common._norm_id(json.loads(line)["id"]) for line in f if line.strip()}
    if set(ids) & train:
        raise ValueError("Clean test manifest overlaps the model's training IDs")
    return ids


def _paired(preds, meta, dim, gating, ids=None):
    """Return (pred_vec, human_vec) for one dim over the given id subset (default: all preds)."""
    P, G = [], []
    keys = ids if ids is not None else preds.keys()
    for rid in keys:
        pr = preds.get(rid)
        if not pr or dim not in pr or rid not in meta:
            continue
        g = common.human_mean(meta[rid], dim, gating=gating)
        if g is None:
            continue
        P.append(pr[dim])
        G.append(g)
    return P, G


def pooled_by_dim(preds, meta, ids=None, gating="mask"):
    """{dim -> (r, n)} plus ('__avg__', mean-over-dims)."""
    out = {}
    cors = []
    for dim in common.REALISM_DIMS:
        P, G = _paired(preds, meta, dim, gating, ids)
        r = common.pearson(P, G)
        out[dim] = (r, len(P))
        cors.append(r)
    out["__avg__"] = (float(np.nanmean(cors)), None)
    return out


def within_speaker_avg(preds, meta, ids, gating="mask"):
    """Mean-over-dims of within-speaker Pearson, averaged over speakers (equal speaker weight).

    Returns (mean_within_speaker_r, {speaker -> avg_r_over_dims})."""
    by_spk = collections.defaultdict(list)
    for rid in ids:
        by_spk[speaker_of(rid)].append(rid)
    per_spk = {}
    for spk, sids in sorted(by_spk.items()):
        if len(sids) < 3:                       # e.g. pro-mufei has 2 -> not estimable
            continue
        cors = []
        for dim in common.REALISM_DIMS:
            P, G = _paired(preds, meta, dim, gating, sids)
            cors.append(common.pearson(P, G))
        per_spk[spk] = float(np.nanmean(cors))
    mean_ws = float(np.nanmean(list(per_spk.values()))) if per_spk else float("nan")
    return mean_ws, per_spk


def _fmt(r):
    return "  nan" if r != r else f"{r:+.3f}"


def main():
    meta = common.load_metadata(CLEAN_SPLIT)
    v2_ids = load_v2_ids()
    spk_counts = collections.Counter(speaker_of(i) for i in v2_ids)

    clean_dir = common.DRAME_DIRS[CLEAN_SPLIT]
    print("=" * 78)
    print("CLEAN real-rec evaluation on test_real_v2 (held-out) — job-pqa5dxofkv")
    print("=" * 78)
    print(f"clean preds dir : {clean_dir}")
    print(f"test_real_v2    : {len(v2_ids)} ids  {dict(spk_counts)}")

    if not os.path.isdir(clean_dir) or not any(
        os.path.exists(os.path.join(clean_dir, f"role_eval_{common.REL2PRED[d]}_exact.json"))
        for d in common.REALISM_DIMS
    ):
        print("\n[pending] Clean prediction dir not populated yet — the job has not written")
        print("          any role_eval_<dim>_exact.json files. Re-run once job-pqa5dxofkv")
        print("          finishes. (Wiring validated against the contaminated dir separately.)")
        return

    clean = common.load_drame_preds(CLEAN_SPLIT)
    clean = {rid: row for rid, row in clean.items() if rid in set(v2_ids)}
    got = collections.Counter(speaker_of(i) for i in clean)
    n_dims = len({
        d for d in common.REALISM_DIMS
        if os.path.exists(os.path.join(clean_dir, f"role_eval_{common.REL2PRED[d]}_exact.json"))
    })
    print(f"clean preds     : {len(clean)} ids, {n_dims}/10 dims present  {dict(got)}")
    if n_dims < 10:
        print("          [partial] not all 10 dims written yet — numbers below are provisional.")

    # (1) HEADLINE ------------------------------------------------------------------
    print("\n(1) HEADLINE — pooled Pearson, all held-out (impute gating = paper convention)")
    imp = pooled_by_dim(clean, meta, ids=v2_ids, gating="impute")
    for dim in common.REALISM_DIMS:
        r, n = imp[dim]
        print(f"    {common.REALISM_LABELS[dim]:10s} {dim:24s} n={n:5d}  r={_fmt(r)}")
    print(f"    {'OVERALL':10s} {'(mean over 10 dims)':24s}          r={_fmt(imp['__avg__'][0])}"
          f"   [paper contaminated: +0.247]")
    msk = pooled_by_dim(clean, meta, ids=v2_ids, gating="mask")
    print(f"    {'OVERALL':10s} {'(mask gating, conservative)':24s}    r={_fmt(msk['__avg__'][0])}")

    # (2) OVERLAP cross-check -------------------------------------------------------
    print("\n(2) OVERLAP cross-check — 544 ids shared with contaminated dir (zrr+qyf+2)")
    contam = common.load_drame_preds(CONTAM_SPLIT)
    overlap = sorted(set(clean) & set(contam))
    print(f"    overlap ids: {len(overlap)}  {dict(collections.Counter(speaker_of(i) for i in overlap))}")
    # prediction agreement (same model+audio => should be ~identical)
    diffs = [abs(clean[i][d] - contam[i][d])
             for i in overlap for d in common.REALISM_DIMS
             if d in clean[i] and d in contam[i]]
    if diffs:
        print(f"    pred |Δ| over shared cells: mean={np.mean(diffs):.4f} max={np.max(diffs):.4f}"
              f"  (near 0 ⇒ same checkpoint)")
    r_clean_ov = pooled_by_dim(clean, meta, ids=overlap)["__avg__"][0]
    r_contam_ov = pooled_by_dim(contam, meta, ids=overlap)["__avg__"][0]
    print(f"    overall r on overlap — clean dir : {_fmt(r_clean_ov)}")
    print(f"    overall r on overlap — contam dir: {_fmt(r_contam_ov)}   [interim held-out ref: +0.049]")

    # (3) NEW-ONLY ------------------------------------------------------------------
    new_ids = sorted(set(clean) - set(contam))     # yy + zyx (540)
    print(f"\n(3) NEW-ONLY — {len(new_ids)} previously-unscored held-out samples "
          f"{dict(collections.Counter(speaker_of(i) for i in new_ids))}")
    print(f"    overall r (impute): {_fmt(pooled_by_dim(clean, meta, ids=new_ids)['__avg__'][0])}")

    # (4) PER-SPEAKER within-speaker decomposition ----------------------------------
    print("\n(4) PER-SPEAKER within-speaker decomposition (E2)")
    mean_ws, per_spk = within_speaker_avg(clean, meta, list(clean.keys()))
    for spk, r in sorted(per_spk.items()):
        print(f"    within-{spk:10s} avg-over-dims r={_fmt(r)}")
    print(f"    mean-within-speaker r : {_fmt(mean_ws)}")
    print(f"    pooled r (from §1)    : {_fmt(imp['__avg__'][0])}   "
          f"(pooled − within = across-speaker mean-shift inflation)")
    print(f"    human pooled single-rater reference: +{HUMAN_CEILING_REF:.3f}")

    # persist -----------------------------------------------------------------------
    os.makedirs(common.OUT_DIR, exist_ok=True)
    out_path = os.path.join(common.OUT_DIR, "realrec_v2_clean_summary.json")
    json.dump({
        "n_ids": len(clean), "n_dims_present": n_dims,
        "pooled_impute": {d: imp[d][0] for d in common.REALISM_DIMS},
        "overall_impute": imp["__avg__"][0], "overall_mask": msk["__avg__"][0],
        "overlap_r_clean": r_clean_ov, "overlap_r_contam": r_contam_ov,
        "new_only_r": pooled_by_dim(clean, meta, ids=new_ids)["__avg__"][0],
        "within_speaker_mean": mean_ws, "within_speaker_by_spk": per_spk,
    }, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
