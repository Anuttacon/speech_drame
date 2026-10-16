"""
E11 text-only analysis — compare full DRAME-Eval (audio) vs TEXT-ONLY judge (Qwen2.5-7B on
oracle transcript+profile+scene, job-niucor0rdv) vs AUDIO-ABLATED DRAME (from exp_e11_ablation).
Answers gEEq G1.2: how much of the signal is available from text alone (oracle-ASR upper bound).

Run AFTER the text-judge job lands. Reuses mask gating + per-dim Pearson.
Writes analysis/out/e11_textonly.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids
from exp_e1_bootstrap import load_dir_preds, realism_human, build_cells, fmt
from exp_e11_ablation import PROSODY_EMO, CONSIST_CTX, sem_stratum

TEXT = {"test_base": "results_textjudge/test_all", "test_real_v2": "results_textjudge/test_real_v2"}
FULL = {"test_base": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall",
        "test_real_v2": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2"}
ABL = {"test_base": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall_zeroaudio",
       "test_real_v2": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2_zeroaudio"}


def perdim(preds, meta, dims, idf=None):
    cells = build_cells(preds, meta, dims, realism_human, id_filter=idf)
    return {d: common.pearson(*cells[d][1:]) for d in dims}


def gavg(pd, group):
    vs = [pd[d] for d in group if pd.get(d) == pd.get(d)]
    return float(np.mean(vs)) if vs else float("nan")


def main():
    if not os.path.exists(os.path.join(common.EVAL_DIR, TEXT["test_base"], "role_eval_pitch_dynamics_exact.json")):
        print("Text-judge preds not present yet — waiting on job-niucor0rdv."); return
    out = {}
    md = ["# E11 text-only judge vs full / audio-ablated DRAME", "",
          "Text-only = Qwen2.5-7B-Instruct on oracle transcript + profile + scene (NO audio), "
          "same rubric + expectation decoding. Oracle-ASR upper bound for a text shortcut. "
          "Per-dim Pearson (mask) vs human.", ""]
    for split in ("test_base", "test_real_v2"):
        meta = common.load_metadata("realism_base" if split == "test_base" else "realism_real")
        idf = None if split == "test_base" else set(load_v2_ids())
        pf = perdim(load_dir_preds(FULL[split], common.REALISM_DIMS), meta, common.REALISM_DIMS, idf)
        pt = perdim(load_dir_preds(TEXT[split], common.REALISM_DIMS), meta, common.REALISM_DIMS, idf)
        pa = perdim(load_dir_preds(ABL[split], common.REALISM_DIMS), meta, common.REALISM_DIMS, idf) \
            if os.path.exists(os.path.join(common.EVAL_DIR, ABL[split], "role_eval_pitch_dynamics_exact.json")) else None
        md += [f"## {split}", "", "| Dim group | full DRAME (audio) | text-only | audio-ablated |", "|---|---|---|---|"]
        for gname, group in [("prosody/emotion", PROSODY_EMO), ("consistency/context", CONSIST_CTX)]:
            aval = fmt(gavg(pa, group)) if pa else "—"
            md.append(f"| {gname} | {fmt(gavg(pf, group))} | {fmt(gavg(pt, group))} | {aval} |")
        allf = float(np.nanmean([pf[d] for d in common.REALISM_DIMS]))
        allt = float(np.nanmean([pt[d] for d in common.REALISM_DIMS]))
        alla = float(np.nanmean([pa[d] for d in common.REALISM_DIMS])) if pa else float("nan")
        md.append(f"| **all 10** | **{fmt(allf)}** | **{fmt(allt)}** | **{fmt(alla)}** |")
        md.append("")
        out[split] = {"full": allf, "textonly": allt, "ablated": alla,
                      "text_prosody": gavg(pt, PROSODY_EMO), "text_consist": gavg(pt, CONSIST_CTX)}

    # sm stratum on test_base consistency dims
    meta_b = common.load_metadata("realism_base")
    pf_dir = load_dir_preds(FULL["test_base"], common.REALISM_DIMS)
    pt_dir = load_dir_preds(TEXT["test_base"], common.REALISM_DIMS)
    md += ["## test_base consistency/context by semantic_match stratum", "",
           "| sm stratum | full DRAME | text-only |", "|---|---|---|"]
    out["strata"] = {}
    for name, lo, hi in [("sm≥4 (text-invisible)", 4, 99), ("3≤sm<4", 3, 4), ("sm<3 (text-detectable)", 0, 3)]:
        ids = sem_stratum(meta_b, lo, hi)
        gf = gavg(perdim(pf_dir, meta_b, CONSIST_CTX, ids), CONSIST_CTX)
        gt = gavg(perdim(pt_dir, meta_b, CONSIST_CTX, ids), CONSIST_CTX)
        md.append(f"| {name} (n={len(ids)}) | {fmt(gf)} | {fmt(gt)} |")
        out["strata"][name] = {"full": gf, "textonly": gt}
    md += ["", "_Expected: text-only ≈0 on prosody/emotion (not inferable from text) and on "
           "the sm≥4 stratum (content matches → no textual tell); it may recover some "
           "consistency signal on sm<3 (content-detectable negatives — humans use the same "
           "text). The gap between full-DRAME and text-only on prosody/emotion is the "
           "audio-specific contribution the benchmark uniquely measures._", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e11_textonly.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e11_textonly.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e11_textonly.md"))
    for s in ("test_base", "test_real_v2"):
        o = out[s]; print(f"  {s}: full {fmt(o['full'])} text-only {fmt(o['textonly'])} ablated {fmt(o['ablated'])}")


if __name__ == "__main__":
    main()
