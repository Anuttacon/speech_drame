"""
E10 — Qualitative error analysis (reviewer NhWo non-critical, but strengthens the
synthetic-to-real-gap story). Surfaces where DRAME-Eval disagrees most with humans on the
held-out real recordings, restricted to LOW-AMBIGUITY samples (raters agree, so the target
is well-defined and the model is simply wrong) plus a couple of successes for contrast.

Focus dims = voice_identity_matching, traits_embodiment (the two real-rec dims where
correlation collapses to ~0), now that the clean test_real_v2 re-inference has landed.

For each example we emit the transcript, profile, scene, human mean, rater spread, model
prediction, |error|, and a rule-based one-line diagnosis (over/under-rating + agreement).
No audio re-processing; everything from released metadata + saved predictions.

Run from analysis/:  PYTHONPATH=. python3 exp_e10_error_analysis.py
Writes analysis/out/e10_error_analysis.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids, speaker_of
from exp_e1_bootstrap import load_dir_preds

DRAME_REAL_V2_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2"
FOCUS_DIMS = ["voice_identity_matching", "traits_embodiment", "local_scene_fit", "global_story_coherence"]
SPREAD_MAX = 0.6   # rater std below this = low-ambiguity (raters agree)
TOPK = 5


def rater_spread(rec, dim):
    arr = common.human_rater_arrays(rec, dim, gating="mask")
    return (float(np.std(arr)), len(arr)) if len(arr) >= 2 else (float("nan"), len(arr))


def shorten(s, n=180):
    if not s:
        return ""
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def diagnose(pred, human, spread):
    direction = "over-rates" if pred > human else "under-rates"
    agree = "high agreement" if spread < 0.4 else "moderate agreement"
    return f"model {direction} by {abs(pred - human):.2f} on a {agree} sample (rater σ={spread:.2f})"


def collect(meta, preds, v2, dim):
    rows = []
    for rid, pr in preds.items():
        if dim not in pr or rid not in meta or rid not in v2:
            continue
        rec = meta[rid]
        h = common.human_mean(rec, dim, gating="mask")
        if h is None:
            continue
        sp, nr = rater_spread(rec, dim)
        if sp != sp or sp > SPREAD_MAX:
            continue
        rows.append({"id": rid, "speaker": speaker_of(rid), "pred": pr[dim], "human": h,
                     "err": abs(pr[dim] - h), "spread": sp, "n_raters": nr,
                     "transcript": shorten(rec.get("transcript")),
                     "profile": shorten(rec.get("profile")),
                     "scene": shorten((rec.get("scene") or {}).get("local")),
                     "char_style": rec.get("char_style")})
    return rows


def render_examples(title, rows, note=""):
    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    for i, r in enumerate(rows, 1):
        lines += [f"**{i}. `{r['id']}`** (speaker {r['speaker']})  —  human **{r['human']:.2f}** "
                  f"vs model **{r['pred']:.2f}**  (|err| {r['err']:.2f}, rater σ {r['spread']:.2f}, "
                  f"{r['n_raters']} raters)",
                  f"- diagnosis: {diagnose(r['pred'], r['human'], r['spread'])}",
                  f"- transcript: {r['transcript']}",
                  f"- profile: {r['profile']}",
                  f"- char_style: {r['char_style']}", ""]
    return lines


def main():
    meta = common.load_metadata("realism_real")
    v2 = set(load_v2_ids())
    preds = load_dir_preds(DRAME_REAL_V2_DIR, common.REALISM_DIMS)
    present = [d for d in common.REALISM_DIMS if any(d in pr for pr in preds.values())]
    out = {"focus_dims": FOCUS_DIMS, "spread_max": SPREAD_MAX, "n_dims_present": len(present)}
    md = ["# E10 — Qualitative error analysis (held-out real recordings, test_real_v2)", "",
          f"DRAME-Eval on the clean test_real_v2 (1,084). Examples restricted to "
          f"low-ambiguity samples (rater σ ≤ {SPREAD_MAX}) so the human target is "
          f"well-defined. All {len(present)}/10 dimension predictions available.", ""]

    for dim in FOCUS_DIMS:
        if dim not in present:
            md += [f"## {common.REALISM_LABELS[dim]} — _pending re-inference_", ""]
            continue
        rows = collect(meta, preds, v2, dim)
        if not rows:
            continue
        rows.sort(key=lambda r: -r["err"])
        worst = rows[:TOPK]
        best = sorted(rows, key=lambda r: r["err"])[:2]
        r_all = common.pearson([r["pred"] for r in rows], [r["human"] for r in rows])
        md += render_examples(
            f"{common.REALISM_LABELS[dim]} ({dim}) — largest disagreements "
            f"(within-dim r={r_all:+.3f} on {len(rows)} low-ambiguity samples)",
            worst,
            note="_Cases where humans agree but the model is far off — the clearest failure "
                 "signatures._")
        md += render_examples(f"{common.REALISM_LABELS[dim]} — closest agreements (successes)", best)
        out[dim] = {"n_low_ambiguity": len(rows), "within_dim_r": r_all,
                    "worst": [{k: r[k] for k in ("id", "speaker", "pred", "human", "err", "spread")} for r in worst],
                    "best": [{k: r[k] for k in ("id", "speaker", "pred", "human", "err", "spread")} for r in best]}

    # aggregate error direction summary
    md += ["## Aggregate error-direction summary (low-ambiguity samples)", "",
           "| Dimension | n | mean signed err (pred−human) | mean |err| | within-dim r |",
           "|---|---|---|---|---|"]
    for dim in FOCUS_DIMS:
        if dim not in present:
            md.append(f"| {common.REALISM_LABELS[dim]} | _pending_ | — | — | — |"); continue
        rows = collect(meta, preds, v2, dim)
        if not rows:
            continue
        signed = np.mean([r["pred"] - r["human"] for r in rows])
        absmean = np.mean([r["err"] for r in rows])
        r_all = common.pearson([r["pred"] for r in rows], [r["human"] for r in rows])
        md.append(f"| {common.REALISM_LABELS[dim]} | {len(rows)} | {signed:+.3f} | {absmean:.3f} | {r_all:+.3f} |")
    md += ["", "_A consistent negative signed error would mean the model systematically "
           "under-rates real recordings (the synthetic-to-real gap manifesting as pessimism); "
           "near-zero within-dim r confirms the disagreements are unstructured, not a fixable "
           "offset._", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e10_error_analysis.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e10_error_analysis.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e10_error_analysis.md"))
    print("\n--- E10 HIGHLIGHTS ---")
    for dim in FOCUS_DIMS:
        if dim in out:
            print(f"  {common.REALISM_LABELS[dim]:10s} within-dim r {out[dim]['within_dim_r']:+.3f} "
                  f"(n_low_ambiguity {out[dim]['n_low_ambiguity']})")


if __name__ == "__main__":
    main()
