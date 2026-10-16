#!/usr/bin/env python3
"""E2b — WITHIN-SPEAKER human single-rater agreement on the clean real-rec set.

Reviewer (6th pass): the paper contrasts POOLED single-rater human agreement (0.561)
with the model's WITHIN-speaker correlation (0.029) — different protocols. This computes
the human single-rater-vs-consensus Pearson *within each held-out speaker*, averaged over
speakers, i.e. the exact analog of the model's within_speaker_avg (realrec_v2_clean.py).

For each of the 4 held-out speakers, for each dimension: gather (held-out rater, mean-of-rest)
pairs restricted to that speaker's utterances, Pearson(H,R); mean over dims -> per-speaker;
then average over speakers (equal speaker weight). Reports both the within-speaker human
reference and, for comparison, the pooled single-vs-rest (which is the 0.561 in the paper).
"""
import collections
import json
import os
import numpy as np
import common as C
from realrec_v2_clean import load_v2_ids, speaker_of

SPLIT = "realism_real_v2"
GATING = "mask"
DIMS = C.REALISM_DIMS


def single_vs_rest_pairs(recs, dim, gating=GATING):
    H, R = [], []
    for rec in recs:
        arr = C.human_rater_arrays(rec, dim, gating=gating)
        if len(arr) < 2:
            continue
        a = np.array(arr, float)
        for i in range(len(a)):
            H.append(a[i]); R.append((a.sum() - a[i]) / (len(a) - 1))
    return np.array(H), np.array(R)


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    meta = C.load_metadata(SPLIT)
    v2 = load_v2_ids()
    recs_by_spk = collections.defaultdict(list)
    for rid in v2:
        rec = meta.get(rid)
        if rec is not None:
            recs_by_spk[speaker_of(rid)].append(rec)

    print("held-out speakers:", {k: len(v) for k, v in recs_by_spk.items()})

    # --- within-speaker: per speaker, mean-over-dims single-vs-rest Pearson ---
    per_spk = {}
    for spk, recs in recs_by_spk.items():
        # Match the model protocol: the two-utterance speaker is not estimable.
        if len(recs) < 3:
            continue
        dim_rs = []
        for dim in DIMS:
            H, R = single_vs_rest_pairs(recs, dim)
            if len(H) >= 2 and np.std(H) > 0 and np.std(R) > 0:
                dim_rs.append(float(np.corrcoef(H, R)[0, 1]))
        per_spk[spk] = float(np.nanmean(dim_rs)) if dim_rs else float("nan")
    within_human = float(np.nanmean(list(per_spk.values())))

    # --- pooled (paper's 0.561): single-vs-rest pooled over all speakers/samples ---
    all_recs = [r for recs in recs_by_spk.values() for r in recs]
    pooled_rs = []
    for dim in DIMS:
        H, R = single_vs_rest_pairs(all_recs, dim)
        if len(H) >= 2 and np.std(H) > 0 and np.std(R) > 0:
            pooled_rs.append(float(np.corrcoef(H, R)[0, 1]))
    pooled_human = float(np.nanmean(pooled_rs))

    print("\nWITHIN-speaker human single-rater agreement (per speaker, mean-over-dims):")
    for spk, r in sorted(per_spk.items()):
        print(f"  {spk:14s} {r:+.3f}")
    print(f"  mean-within-speaker HUMAN = {within_human:+.3f}")
    print(f"  pooled single-vs-rest HUMAN = {pooled_human:+.3f}  (paper's 0.561 analog)")
    print(f"  (model within-speaker r for reference = +0.029; pooled model = -0.142)")

    out = {"within_speaker_human": within_human, "pooled_human": pooled_human,
           "per_speaker": per_spk, "n_speakers": len(per_spk)}
    json.dump(out, open(os.path.join(C.OUT_DIR, "e2b_human_within_speaker.json"), "w"), indent=2)
    md = ["# E2b — Within-speaker human single-rater agreement (clean real-rec, mask)", "",
          f"- **Within-speaker** human single-rater-vs-consensus (per speaker, mean over 10 dims, "
          f"averaged over the {len(per_spk)} held-out speakers): **{within_human:+.3f}**",
          f"- Pooled single-vs-rest human (the paper's headline 0.561 analog): **{pooled_human:+.3f}**",
          f"- Model within-speaker r = +0.029; pooled model r = −0.142.", "",
          "| Speaker | n | within-speaker single-rater r |", "|---|---|---|"]
    for spk, r in sorted(per_spk.items()):
        md.append(f"| {spk} | {len(recs_by_spk[spk])} | {r:+.3f} |")
    open(os.path.join(C.OUT_DIR, "e2b_human_within_speaker.md"), "w").write("\n".join(md) + "\n")
    print("\nwrote out/e2b_human_within_speaker.{json,md}")


if __name__ == "__main__":
    main()
