"""
Causal contrast: does SPEAKER OVERLAP explain the seen-vs-unseen swing?

Compares three LoRA runs on held-out real-rec clips, whole-set Pearson (per-dim over ids,
avg over 10 dims — the paper's 0.247 convention):

  A. original all-real-v2 (cross-speaker, UNSEEN):  -0.142   [train 1080 / 10k steps / DeepSpeed]
  B. matched DISJOINT control (cross-speaker, UNSEEN, this run):  train 1624 / 2000 / DDP
  C. per-speaker SEEN run (speakers seen, unseen clips): +0.112 [train 1732 / 2000 / DDP]

B matches C on train-size/steps/recipe and differs ONLY in speaker geometry (disjoint vs
overlap). So (C - B) isolates the speaker-overlap effect under matched conditions; (A ~ B)
confirms the recipe change didn't create the cross-speaker number.

Tightest sub-contrast: yy+zyx are the held-out TEST speakers of B (all 540 unseen) and were
also the amateurs in C's test (seen there). Comparing yy+zyx across C (seen) and B (unseen)
holds the speakers fixed.

Run from analysis/ after the disjoint eval populates results:  python3 realrec_disjoint_contrast.py [ckpt]
"""
from __future__ import annotations
import collections, json, os, sys
import numpy as np
import common

if len(sys.argv) > 1:  # override disjoint checkpoint dir
    common.DRAME_DIRS["disjoint_ctrl"] = os.path.join(
        common.EVAL_DIR, f"results/sft_lora_disjoint_real_v2_checkpoint-{sys.argv[1]}_test_disjoint")

XSPK = "realism_real_v2"     # A: original cross-speaker (-0.142)
SEEN = "perspk_seen"         # C: seen-speaker run
DISJ = "disjoint_ctrl"       # B: matched disjoint control


def speaker_of(rid):
    p = rid.replace("role-eval_", "").split("_")
    return p[1] if p and p[0].startswith("v") and p[0][1:].isdigit() else p[0]


def overall(preds, meta, ids, fn=None):
    fn = fn or common.pearson
    rs = []
    for d in common.REALISM_DIMS:
        P, G = [], []
        for rid in ids:
            pr = preds.get(rid)
            if not pr or d not in pr or rid not in meta:
                continue
            g = common.human_mean(meta[rid], d, gating="impute")
            if g is None:
                continue
            P.append(pr[d]); G.append(g)
        rs.append(fn(np.array(P, float), np.array(G, float)))
    return float(np.nanmean(rs))


def f(r):
    return " nan" if r != r else f"{r:+.3f}"


def main():
    dj_dir = common.DRAME_DIRS[DISJ]
    if not os.path.isdir(dj_dir) or not any(
        os.path.exists(os.path.join(dj_dir, f"role_eval_{common.REL2PRED[d]}_exact.json"))
        for d in common.REALISM_DIMS):
        print("[pending] disjoint eval preds not populated:", dj_dir); return

    meta = common.load_metadata(DISJ)
    dj = common.load_drame_preds(DISJ)
    seen = common.load_drame_preds(SEEN)
    xr_preds = common.load_drame_preds(XSPK); xr_meta = common.load_metadata(XSPK)

    dj_ids = [i for i in dj if i in meta]
    seen_ids = [i for i in seen if i in meta]
    A = overall(xr_preds, xr_meta, [i for i in xr_preds if i in xr_meta])
    B = overall(dj, meta, dj_ids)
    C = overall(seen, meta, seen_ids)

    print("=" * 78)
    print("CAUSAL CONTRAST — is the seen-vs-unseen swing due to SPEAKER OVERLAP?")
    print("=" * 78)
    print("  A. original cross-speaker (UNSEEN, 1080/10k/DeepSpeed)   : %s   [-0.142 headline]" % f(A))
    print("  B. matched DISJOINT control (UNSEEN, 1624/2000/DDP) n=%d : %s" % (len(dj_ids), f(B)))
    print("  C. per-speaker SEEN run (SEEN, 1732/2000/DDP)       n=%d : %s" % (len(seen_ids), f(C)))
    print("  --------------------------------------------------------------------------")
    print("  C - B (speaker-overlap effect, matched recipe): %s" % f(C - B))
    print("  A vs B (recipe-change check; should be similar): %s vs %s" % (f(A), f(B)))

    # tightest: yy+zyx held out in BOTH; seen in C, unseen in B
    print("\n  yy+zyx sub-contrast (same speakers, seen in C vs unseen in B):")
    for spk in ("yy", "zyx"):
        b_ids = [i for i in dj_ids if speaker_of(i) == spk]
        c_ids = [i for i in seen_ids if speaker_of(i) == spk]
        print("    %-4s : SEEN(C) %s [n=%d]   UNSEEN(B) %s [n=%d]" %
              (spk, f(overall(seen, meta, c_ids)), len(c_ids), f(overall(dj, meta, b_ids)), len(b_ids)))

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump({"A_orig_unseen": A, "B_matched_disjoint_unseen": B, "C_seen": C,
               "speaker_overlap_effect_CmB": C - B, "n_disjoint": len(dj_ids)},
              open(os.path.join(common.OUT_DIR, "realrec_disjoint_contrast.json"), "w"), indent=2)
    print("\nwrote", os.path.join(common.OUT_DIR, "realrec_disjoint_contrast.json"))


if __name__ == "__main__":
    main()
