"""
E4 — Per-domain / per-source breakdowns (reviewer NhWo C1, gEEq "results by source type").

  * Archetype by role_category (7 domains), mask, DRAME + baselines, avg-of-3-dims
  * Realism base by source family (ncssd / STARRAIL / GENSHIN = zh; TV = en) and by
    sample_type (positive / role-neg / context-neg), mask, avg-of-10
  * Real-recording by speaker (baselines; DRAME pending re-inference)

All mask gating (consistent with E1/E5 recommendation). Run from analysis/:
  PYTHONPATH=. python3 exp_e4_domain.py   -> out/e4_domain.{json,md}
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from exp_e1_bootstrap import (
    ARCHE_DRAME_DIR, load_dir_preds, realism_models, build_cells, avg_ci, fmt,
)
from exp_e5_gating import realism_mask, arche_mask
from realrec_v2_clean import load_v2_ids, speaker_of

N_BOOT = 1000
DRAME_BASE_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


def source_family(rid, rec):
    oid = str(rec.get("original_id") or "")
    if oid.startswith("ncssd"):
        return "ncssd(zh)"
    if oid.startswith("STARRAIL"):
        return "STARRAIL(zh)"
    if oid.startswith("GENSHIN"):
        return "GENSHIN(zh)"
    return "TV(en)"


def sample_group(rid, rec):
    if rec.get("sample_type") == "positive":
        return "positive"
    return f"{rec.get('negative_type')}-neg"


def group_avg_table(preds, meta, dims, human_fn, grouper, order=None):
    """{group -> (avg, lo, hi, n)} over dims, mask gating. grouper(rid, rec) -> key."""
    groups = collections.defaultdict(set)
    for rid, rec in meta.items():
        groups[grouper(rid, rec)].add(rid)
    res = {}
    keys = order or sorted(groups)
    for g in keys:
        if g not in groups:
            continue
        cells = build_cells(preds, meta, dims, human_fn, id_filter=groups[g])
        ap, lo, hi = avg_ci(cells, dims, n=N_BOOT)
        n = max((len(cells[d][0]) for d in dims), default=0)
        res[g] = (ap, lo, hi, n)
    return res


def main():
    out = {"n_boot": N_BOOT}
    md = ["# E4 — Per-domain / per-source breakdowns (mask gating)", ""]

    # ---------- Archetype by role_category ----------
    print("E4 archetype by role_category ...")
    meta_a = common.load_metadata("archetype")
    adims = common.ARCHE_DIMS
    RC_ORDER = ["social_identity", "personality_trait", "occupation_daily", "relative_role",
                "events", "occupation_fantasy", "named_character"]
    arows = [("DRAME-Eval", load_dir_preds(ARCHE_DRAME_DIR, adims, is_arche=True))]
    for m in ["Gemini2.5Pro", "GPT4o-audio"]:
        p = common.load_judge_preds_arche(m, "zero")
        if p:
            arows.append((m + " (ZS)", p))
    md += ["## Archetype — Avg(AQ,HL,AP) by role_category (mask)", "",
           "| Model | " + " | ".join(RC_ORDER) + " |", "|" + "---|" * (len(RC_ORDER) + 1)]
    out["archetype_domain"] = {}
    for label, preds in arows:
        res = group_avg_table(preds, meta_a, adims, arche_mask, lambda rid, r: r.get("role_category"), RC_ORDER)
        md.append(f"| {label} | " + " | ".join(f"{fmt(res[g][0])} (n={res[g][3]})" for g in RC_ORDER if g in res) + " |")
        out["archetype_domain"][label] = {g: res[g][0] for g in res}
    md.append("")

    # ---------- Realism base by source family ----------
    print("E4 realism base by source family ...")
    meta_b = common.load_metadata("realism_base")
    dims = common.REALISM_DIMS
    rrows = [("DRAME-Eval", load_dir_preds(DRAME_BASE_DIR, dims))]
    for label, cfg, preds in realism_models("realism_base"):
        rrows.append((label, preds))
    SRC_ORDER = ["ncssd(zh)", "STARRAIL(zh)", "GENSHIN(zh)", "TV(en)"]
    md += ["## Realism base — Avg(10 dims) by source family (mask)", "",
           "_zh families vs the English TV family — the source cut is ~collinear with language._", "",
           "| Model | " + " | ".join(SRC_ORDER) + " |", "|" + "---|" * (len(SRC_ORDER) + 1)]
    out["realism_source"] = {}
    for label, preds in rrows:
        res = group_avg_table(preds, meta_b, dims, realism_mask, source_family, SRC_ORDER)
        md.append(f"| {label} | " + " | ".join(f"{fmt(res[g][0])} (n={res[g][3]})" for g in SRC_ORDER if g in res) + " |")
        out["realism_source"][label] = {g: res[g][0] for g in res}
    md.append("")

    # ---------- Realism base by sample_type ----------
    print("E4 realism base by sample_type ...")
    ST_ORDER = ["positive", "role-neg", "context-neg"]
    md += ["## Realism base — Avg(10 dims) by sample type (mask)", "",
           "_Within-group correlation; positives n=210 so CIs are wide._", "",
           "| Model | " + " | ".join(ST_ORDER) + " |", "|" + "---|" * (len(ST_ORDER) + 1)]
    out["realism_sampletype"] = {}
    for label, preds in rrows:
        res = group_avg_table(preds, meta_b, dims, realism_mask, sample_group, ST_ORDER)
        md.append(f"| {label} | " + " | ".join(f"{fmt(res[g][0])} (n={res[g][3]})" for g in ST_ORDER if g in res) + " |")
        out["realism_sampletype"][label] = {g: res[g][0] for g in res}
    md.append("")

    # ---------- Real-rec by speaker (baselines; DRAME pending) ----------
    print("E4 real-rec by speaker ...")
    v2 = set(load_v2_ids())
    meta_r = common.load_metadata("realism_real")
    SPK_ORDER = ["qyf", "zrr", "zyx", "yy"]  # pro-mufei has 2 -> skip
    brows = []
    for m in common.ZS_STEMS:
        p = common.load_judge_preds(m, "realism_real", "zero")
        if p:
            brows.append((m + " (ZS)", p))
    for m in common.FS_STEMS:
        p = common.load_judge_preds(m, "realism_real", "few")
        if p:
            brows.append((m + " (FS)", p))
    md += ["## Real-recording (test_real_v2) — Avg(10 dims) by speaker (baselines; DRAME pending)", "",
           "_The 4 held-out actors. Per-speaker correlation strips the between-speaker "
           "composition that inflates the pooled number (E2)._", "",
           "| Model | " + " | ".join(SPK_ORDER) + " | pooled |", "|" + "---|" * (len(SPK_ORDER) + 2)]
    out["realrec_speaker"] = {}
    for label, preds in brows:
        pv2 = {rid: v for rid, v in preds.items() if rid in v2}
        res = group_avg_table(pv2, meta_r, dims, realism_mask, lambda rid, r: speaker_of(rid), SPK_ORDER)
        cells_all = build_cells(pv2, meta_r, dims, realism_mask, id_filter=v2)
        pooled = avg_ci(cells_all, dims, n=N_BOOT)[0]
        md.append(f"| {label} | " + " | ".join(f"{fmt(res[g][0])}" for g in SPK_ORDER if g in res)
                  + f" | {fmt(pooled)} |")
        out["realrec_speaker"][label] = {**{g: res[g][0] for g in res}, "pooled": pooled}
    md += ["", "_DRAME-Eval per-speaker + within-speaker decomposition lands with the "
           "540-sample re-inference (E2)._", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e4_domain.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e4_domain.md"), "w").write("\n".join(md))
    print(f"wrote {os.path.join(common.OUT_DIR, 'e4_domain.md')}")


if __name__ == "__main__":
    main()
