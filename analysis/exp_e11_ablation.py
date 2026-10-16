"""
E11 (ablation analysis) — full DRAME-Eval vs AUDIO-ABLATED DRAME-Eval (silence fed instead
of the waveform, `--zero_audio`, job-jlun00yk3d) on test_base + test_real_v2. Any correlation
that SURVIVES with the audio removed is the text/context shortcut inside our own model
(reviewer gEEq G1.2). Run AFTER the ablation job lands.

Per-dim Pearson (mask) vs human, full vs ablated, grouped into {prosody/emotion} (should
collapse to ~0 without audio) vs {consistency/context} (may retain some signal from the
profile/scene text). Also stratified by semantic_match on test_base (sm≥4 text-invisible /
3≤sm<4 / sm<3) — the ablated model should be near-0 on the sm≥4 stratum.

Run from analysis/:  PYTHONPATH=. python3 exp_e11_ablation.py
Writes analysis/out/e11_ablation.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids
from exp_e1_bootstrap import load_dir_preds, realism_human, build_cells, fmt

FULL = {"test_base": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall",
        "test_real_v2": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2"}
ABLATED = {"test_base": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall_zeroaudio",
           "test_real_v2": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2_zeroaudio"}
PROSODY_EMO = ["pitch_variation", "rhythmic_naturalness", "stress_emphasis",
               "emotion_accuracy", "emotion_intensity", "emotion_transition"]
CONSIST_CTX = ["voice_identity_matching", "traits_embodiment", "local_scene_fit", "global_story_coherence"]


def perdim(preds, meta, dims, idf=None):
    cells = build_cells(preds, meta, dims, realism_human, id_filter=idf)
    return {d: common.pearson(*cells[d][1:]) for d in dims}


def group_avg(pd, group):
    vs = [pd[d] for d in group if pd.get(d) == pd.get(d)]
    return float(np.mean(vs)) if vs else float("nan")


def sem_stratum(meta, lo, hi):
    ids = set()
    for rid, r in meta.items():
        v = r.get("annotations", {}).get("semantic_match")
        if v is not None and lo <= float(v) < hi:
            ids.add(rid)
    return ids


def main():
    if not os.path.exists(os.path.join(common.EVAL_DIR, ABLATED["test_base"], "role_eval_pitch_dynamics_exact.json")):
        print("Ablated preds not present yet — waiting on job-jlun00yk3d. Re-run when it lands.")
        return
    out = {}
    md = ["# E11 (ablation) — full vs audio-ablated DRAME-Eval", "",
          "Audio-ablated = silence fed instead of the waveform. Surviving correlation = "
          "text/context shortcut inside the model. Per-dim Pearson (mask) vs human.", ""]

    for split in ("test_base", "test_real_v2"):
        meta = common.load_metadata("realism_base" if split == "test_base" else "realism_real")
        idf = None if split == "test_base" else set(load_v2_ids())
        full = load_dir_preds(FULL[split], common.REALISM_DIMS)
        abl = load_dir_preds(ABLATED[split], common.REALISM_DIMS)
        pf = perdim(full, meta, common.REALISM_DIMS, idf)
        pa = perdim(abl, meta, common.REALISM_DIMS, idf)
        md += [f"## {split}", "",
               "| Dim group | full DRAME | audio-ablated | Δ (ablated−full) |", "|---|---|---|---|"]
        for gname, group in [("prosody/emotion", PROSODY_EMO), ("consistency/context", CONSIST_CTX)]:
            gf, ga = group_avg(pf, group), group_avg(pa, group)
            md.append(f"| {gname} | {fmt(gf)} | {fmt(ga)} | {fmt(ga - gf)} |")
        allf = float(np.nanmean([pf[d] for d in common.REALISM_DIMS]))
        alla = float(np.nanmean([pa[d] for d in common.REALISM_DIMS]))
        md.append(f"| **all 10** | **{fmt(allf)}** | **{fmt(alla)}** | **{fmt(alla-allf)}** |")
        md.append("")
        out[split] = {"full_all": allf, "ablated_all": alla,
                      "prosody_full": group_avg(pf, PROSODY_EMO), "prosody_ablated": group_avg(pa, PROSODY_EMO),
                      "consist_full": group_avg(pf, CONSIST_CTX), "consist_ablated": group_avg(pa, CONSIST_CTX)}

    # semantic_match stratification on test_base consistency dims
    meta_b = common.load_metadata("realism_base")
    full = load_dir_preds(FULL["test_base"], common.REALISM_DIMS)
    abl = load_dir_preds(ABLATED["test_base"], common.REALISM_DIMS)
    md += ["## test_base — consistency/context dims by semantic_match stratum", "",
           "| sm stratum | full DRAME | audio-ablated |", "|---|---|---|"]
    out["strata"] = {}
    for name, lo, hi in [("sm≥4 (text-invisible)", 4, 99), ("3≤sm<4", 3, 4), ("sm<3 (text-detectable)", 0, 3)]:
        ids = sem_stratum(meta_b, lo, hi)
        gf = group_avg(perdim(full, meta_b, CONSIST_CTX, ids), CONSIST_CTX)
        ga = group_avg(perdim(abl, meta_b, CONSIST_CTX, ids), CONSIST_CTX)
        md.append(f"| {name} (n={len(ids)}) | {fmt(gf)} | {fmt(ga)} |")
        out["strata"][name] = {"full": gf, "ablated": ga, "n": len(ids)}
    md += ["", "_Expected: the ablated model collapses on prosody/emotion (no audio → no "
           "signal) and on the sm≥4 stratum (text-invisible); any residual consistency/context "
           "correlation on sm<3 is the profile/scene-text shortcut, which humans use too. "
           "Whatever the split, it quantifies exactly what information the model uses._", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e11_ablation.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e11_ablation.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e11_ablation.md"))
    for split in ("test_base", "test_real_v2"):
        o = out[split]
        print(f"  {split}: full {fmt(o['full_all'])} → ablated {fmt(o['ablated_all'])} "
              f"(prosody {fmt(o['prosody_ablated'])}, consist {fmt(o['consist_ablated'])})")


if __name__ == "__main__":
    main()
