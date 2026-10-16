"""
Seen-speaker experiment analysis: retrained per-speaker LoRA on the 432 held-out
(utterance-disjoint) clips, where EVERY speaker — incl. the professional pro-mufei —
was seen in training. Contrasts with the cross-speaker held-out result (realism_real_v2,
pooled Pearson -0.142) to test whether the model generalizes WITHIN a seen speaker.

Reports (whole-set: per-dim Pearson over all held-out ids, then averaged over 10 dims —
the paper's 0.247 convention), broken out by the groups the user cares about:
  (1) OVERALL held-out (all 432)      : Pearson + Spearman + bootstrap CI
  (2) pro-mufei (professional, SEEN)  vs  amateurs (SEEN)  : the key split
  (3) per-speaker pooled r
  (4) human leave-one-rater-out ceiling, overall and per group
  (5) side-by-side vs the cross-speaker number (realism_real_v2)

Run from analysis/ AFTER the eval job populates the preds dir:  python3 realrec_perspk_eval.py
"""
from __future__ import annotations
import collections, json, os, sys
import numpy as np
import common

# Optional: pass a checkpoint number (e.g. `python3 realrec_perspk_eval.py 1500`) to point
# the perspk_seen preds dir at that checkpoint's eval output instead of the default.
if len(sys.argv) > 1:
    common.DRAME_DIRS["perspk_seen"] = os.path.join(
        common.EVAL_DIR, f"results/sft_lora_perspk_real_v2_checkpoint-{sys.argv[1]}_test_perspk")

SPLIT = "perspk_seen"
XSPK = "realism_real_v2"            # cross-speaker held-out baseline (-0.142)
HELDOUT = os.path.join(common.EVAL_DIR, "data/test_perspk_real_v2/data.jsonl")
PRO = "pro-mufei"


def speaker_of(rid):
    p = rid.replace("role-eval_", "").split("_")
    return p[1] if p and p[0].startswith("v") and p[0][1:].isdigit() else p[0]


def held_ids():
    return [common._norm_id(json.loads(l)["id"]) for l in open(HELDOUT)]


def aligned(preds, meta, dim, ids, gating="impute"):
    P, G = [], []
    for rid in ids:
        pr = preds.get(rid)
        if not pr or dim not in pr or rid not in meta:
            continue
        g = common.human_mean(meta[rid], dim, gating=gating)
        if g is None:
            continue
        P.append(pr[dim]); G.append(g)
    return np.array(P, float), np.array(G, float)


def overall(preds, meta, ids, fn, gating="impute"):
    return float(np.nanmean([fn(*aligned(preds, meta, d, ids, gating)) for d in common.REALISM_DIMS]))


def overall_ci(preds, meta, ids, fn, gating="impute", n=2000, seed=0):
    cols = {d: aligned(preds, meta, d, ids, gating) for d in common.REALISM_DIMS}
    # rebuild id-aligned maps for resampling
    idmap = {}
    for d in common.REALISM_DIMS:
        P, G = [], []
        ii = []
        for rid in ids:
            pr = preds.get(rid)
            if not pr or d not in pr or rid not in meta:
                continue
            g = common.human_mean(meta[rid], d, gating=gating)
            if g is None:
                continue
            P.append(pr[d]); G.append(g); ii.append(rid)
        idmap[d] = (dict(zip(ii, P)), dict(zip(ii, G)))
    keep = sorted(set(ids) & set(preds) & set(meta))
    rng = np.random.default_rng(seed); m = len(keep); stats = []
    for _ in range(n):
        samp = [keep[k] for k in rng.integers(0, m, m)]
        rs = []
        for d in common.REALISM_DIMS:
            pm, gm = idmap[d]
            xs = [pm[i] for i in samp if i in pm]; ys = [gm[i] for i in samp if i in gm]
            rs.append(fn(np.array(xs), np.array(ys)))
        stats.append(np.nanmean(rs))
    stats = np.array(stats); stats = stats[np.isfinite(stats)]
    return float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def human_ceiling(meta, ids, gating="impute"):
    per_dim = []
    for d in common.REALISM_DIMS:
        a, b = [], []
        for rid in ids:
            if rid not in meta:
                continue
            arr = common.human_rater_arrays(meta[rid], d, gating=gating)
            if len(arr) < 2:
                continue
            arr = np.array(arr, float)
            for k in range(len(arr)):
                a.append(arr[k]); b.append(np.delete(arr, k).mean())
        per_dim.append(common.pearson(a, b))
    return float(np.nanmean(per_dim))


def f(r):
    return " nan" if r != r else f"{r:+.3f}"


def main():
    meta = common.load_metadata(SPLIT)
    cdir = common.DRAME_DIRS[SPLIT]
    if not os.path.isdir(cdir) or not any(
        os.path.exists(os.path.join(cdir, f"role_eval_{common.REL2PRED[d]}_exact.json"))
        for d in common.REALISM_DIMS):
        print("[pending] eval preds not populated yet:", cdir); return
    preds = common.load_drame_preds(SPLIT)
    ids = [i for i in held_ids() if i in preds and i in meta]
    pro = [i for i in ids if speaker_of(i) == PRO]
    ama = [i for i in ids if speaker_of(i) != PRO]

    print("=" * 80)
    print("SEEN-SPEAKER experiment — retrained per-speaker LoRA on held-out utterances")
    print("=" * 80)
    print("preds dir:", cdir)
    print("held-out: %d (pro-mufei %d [SEEN professional], amateurs %d [SEEN])" % (len(ids), len(pro), len(ama)))

    rp = overall(preds, meta, ids, common.pearson); rs = overall(preds, meta, ids, common.spearman)
    plo, phi = overall_ci(preds, meta, ids, common.pearson)
    print("\n(1) OVERALL held-out (whole-set, per-dim avg)")
    print("    Pearson  %s  [%s, %s]" % (f(rp), f(plo), f(phi)))
    print("    Spearman %s" % f(rs))

    print("\n(2) KEY SPLIT — professional (seen) vs amateurs (seen)")
    for name, sub in [("pro-mufei (SEEN)", pro), ("amateurs (SEEN)", ama)]:
        print("    %-18s n=%4d  Pearson=%s  Spearman=%s  human_ceiling=%s" %
              (name, len(sub), f(overall(preds, meta, sub, common.pearson)),
               f(overall(preds, meta, sub, common.spearman)), f(human_ceiling(meta, sub))))

    print("\n(3) per-speaker pooled r")
    by = collections.defaultdict(list)
    for i in ids:
        by[speaker_of(i)].append(i)
    for spk in sorted(by):
        print("    %-10s n=%4d  Pearson=%s  Spearman=%s" %
              (spk, len(by[spk]), f(overall(preds, meta, by[spk], common.pearson)),
               f(overall(preds, meta, by[spk], common.spearman))))

    hc = human_ceiling(meta, ids)
    print("\n(4) human ceiling (leave-one-rater-out, overall): %s" % f(hc))

    print("\n(5) CONTRAST vs cross-speaker held-out (realism_real_v2)")
    try:
        xpreds = common.load_drame_preds(XSPK); xmeta = common.load_metadata(XSPK)
        xids = [i for i in xpreds if i in xmeta]
        xr = overall(xpreds, xmeta, xids, common.pearson)
    except Exception as e:
        xr = float("nan")
    print("    cross-speaker (UNSEEN speakers) Pearson : %s   [the -0.142 headline]" % f(xr))
    print("    same-speaker  (SEEN speakers)   Pearson : %s" % f(rp))
    print("    -> seen-minus-unseen generalization gain: %s" % f(rp - xr))

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump({
        "n": len(ids), "overall_pearson": rp, "overall_pearson_ci": [plo, phi],
        "overall_spearman": rs, "pro_pearson": overall(preds, meta, pro, common.pearson),
        "amateur_pearson": overall(preds, meta, ama, common.pearson),
        "human_ceiling": hc, "cross_speaker_pearson": xr,
        "per_speaker": {s: overall(preds, meta, by[s], common.pearson) for s in by},
    }, open(os.path.join(common.OUT_DIR, "realrec_perspk_eval.json"), "w"), indent=2)
    print("\nwrote", os.path.join(common.OUT_DIR, "realrec_perspk_eval.json"))


if __name__ == "__main__":
    main()
