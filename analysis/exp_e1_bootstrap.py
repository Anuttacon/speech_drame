"""Bootstrap CIs, paired comparisons, and contamination diagnostics.

Primary protocol: mask human N/A values; use test_real_v2 for authentic speech;
use the addcontent Instruct-LoRA directory for archetype. All computations read
saved artifacts. Table numbers in this script follow the rebuttal draft.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids, speaker_of

N_BOOT = 2000
SEED = 12345
GATING = "mask"

# Final archetype model; also registered in common.DRAME_DIRS.
ARCHE_DRAME_DIR = os.path.join(
    common.EVAL_DIR, "results_arche/sft_lora_addcontent_specific_dimension_checkpoint-10000"
)
TRAIN_V2 = os.path.join(common.EVAL_DIR, "data/train_all_real_v2/data.jsonl")

# realism base fine-tuning ablation (Tab 11) — dir + published (impute) Avg for reference
ABLATION_BASE = {
    "Base-FT":       ("results/sft_finetune_base_all-real-v2_checkpoint-3000", 0.646),
    "Instruct-FT":   ("results/sft_finetune_all-real-v2_checkpoint-3000", 0.668),
    "Instruct-LoRA": ("results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall", 0.625),
    # Base-LoRA base-test predictions dir unresolved on disk (flagged) — omitted
}
# real-recording ablation (Tab 12) — all on test_real_v2 except the contaminated headline
ABLATION_REAL = {
    "Base-FT":       ("results/sft_finetune_base_all-real-v2_checkpoint-3000_testreal", -0.170, "clean"),
    "Base-LoRA":     ("results/sft_lora_base_all-real-v2_checkpoint-10000_testreal", -0.150, "clean"),
    "Instruct-FT":   ("results/sft_finetune_all-real-v2_checkpoint-3000_testreal", -0.088, "clean"),
    "Instruct-LoRA": ("results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2", -0.142, "clean"),
}
CONTAM_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testreal"


# --------------------------------------------------------------------------------------
# Loaders / ground-truth getters
# --------------------------------------------------------------------------------------
def load_dir_preds(relpath, dims, is_arche=False):
    """Load {id -> {dim: score}} straight from a result dir (bypasses DRAME_DIRS)."""
    d = relpath if os.path.isabs(relpath) else os.path.join(common.EVAL_DIR, relpath)
    out = {}
    for dim in dims:
        fn = f"arche_eval_{dim}_exact.json" if is_arche else f"role_eval_{common.REL2PRED[dim]}_exact.json"
        fp = os.path.join(d, fn)
        if not os.path.exists(fp):
            continue
        for rec in json.load(open(fp)):
            out.setdefault(common._norm_id(rec["id"]), {})[dim] = float(rec["probability_score"])
    return out


def realism_human(rec, dim):
    return common.human_mean(rec, dim, gating=GATING)


def arche_human(rec, dim):
    v = rec.get("annotations", {}).get(dim)
    return float(v) if v is not None else None


# --------------------------------------------------------------------------------------
# Cell arrays + bootstrap machinery
# --------------------------------------------------------------------------------------
def build_cells(preds, meta, dims, human_fn, id_filter=None):
    """{dim -> (ids[np], pred[np], human[np])}, aligned, gating/mask applied via human_fn."""
    cells = {}
    for dim in dims:
        ids, P, G = [], [], []
        for rid, pr in preds.items():
            if dim not in pr or rid not in meta:
                continue
            if id_filter is not None and rid not in id_filter:
                continue
            g = human_fn(meta[rid], dim)
            if g is None or not np.isfinite(g) or not np.isfinite(pr[dim]):
                continue
            ids.append(rid); P.append(pr[dim]); G.append(g)
        cells[dim] = (np.array(ids), np.asarray(P, float), np.asarray(G, float))
    return cells


def cell_ci(cells, dim, n=N_BOOT, seed=SEED):
    _, P, G = cells[dim]
    r = common.pearson(P, G)
    lo, hi = common.paired_pearson_ci(P, G, n=n, seed=seed)
    return r, lo, hi, len(P)


def _union_aligned(cells, dims, id_filter=None):
    """Return (m, [(pred_vec, human_vec) per dim aligned to a shared union index])."""
    union = sorted(set().union(*[set(cells[d][0]) for d in dims])) if dims else []
    if id_filter is not None:
        union = [u for u in union if u in id_filter]
    uidx = {rid: i for i, rid in enumerate(union)}
    m = len(union)
    per_dim = []
    for d in dims:
        ids, P, G = cells[d]
        pv = np.full(m, np.nan); gv = np.full(m, np.nan)
        for r, p, g in zip(ids, P, G):
            if r in uidx:
                pv[uidx[r]] = p; gv[uidx[r]] = g
        per_dim.append((pv, gv))
    return m, per_dim


def _avg_from_aligned(per_dim, samp):
    rs = []
    for pv, gv in per_dim:
        x, y = pv[samp], gv[samp]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            rs.append(np.nan); continue
        rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return np.nanmean(rs) if np.any(np.isfinite(rs)) else np.nan


def avg_ci(cells, dims, n=N_BOOT, seed=SEED):
    """Avg-over-dims Pearson with a shared-id percentile bootstrap."""
    point = float(np.nanmean([common.pearson(*cells[d][1:]) for d in dims]))
    m, per_dim = _union_aligned(cells, dims)
    if m == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.array([_avg_from_aligned(per_dim, rng.integers(0, m, m)) for _ in range(n)])
    stats = stats[np.isfinite(stats)]
    lo, hi = (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))) if len(stats) else (np.nan, np.nan)
    return point, lo, hi


def paired_delta_avg(cellsA, cellsB, dims, n=N_BOOT, seed=SEED):
    """Δ(avg_A − avg_B) on per-dim COMMON ids with shared resample indices.

    Returns (dAvg_point, lo, hi, p_ge0, n_common_union) where p_ge0 = P(Δ<=0) — small p
    means A>B is robust.
    """
    # per-dim common ids
    common_ids = set()
    per_dim = []
    for d in dims:
        idsA, PA, GA = cellsA[d]; idsB, PB, GB = cellsB[d]
        mapA = {r: (p, g) for r, p, g in zip(idsA, PA, GA)}
        mapB = {r: (p, g) for r, p, g in zip(idsB, PB, GB)}
        cids = [r for r in mapA if r in mapB]
        per_dim.append((d, cids, mapA, mapB))
        common_ids.update(cids)
    union = sorted(common_ids)
    uidx = {r: i for i, r in enumerate(union)}
    m = len(union)
    # aligned arrays
    aligned = []
    for d, cids, mapA, mapB in per_dim:
        pa = np.full(m, np.nan); ga = np.full(m, np.nan)
        pb = np.full(m, np.nan); gb = np.full(m, np.nan)
        for r in cids:
            i = uidx[r]
            pa[i], ga[i] = mapA[r]; pb[i], gb[i] = mapB[r]
        aligned.append((pa, ga, pb, gb))

    def avg_side(samp, which):
        rs = []
        for pa, ga, pb, gb in aligned:
            x, y = (pa[samp], ga[samp]) if which == "A" else (pb[samp], gb[samp])
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                rs.append(np.nan); continue
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
        return np.nanmean(rs) if np.any(np.isfinite(rs)) else np.nan

    full = np.arange(m)
    dpoint = avg_side(full, "A") - avg_side(full, "B")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        samp = rng.integers(0, m, m)
        deltas.append(avg_side(samp, "A") - avg_side(samp, "B"))
    deltas = np.array(deltas); deltas = deltas[np.isfinite(deltas)]
    lo, hi = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))) if len(deltas) else (np.nan, np.nan)
    p_ge0 = float(np.mean(deltas <= 0)) if len(deltas) else float("nan")
    return float(dpoint), lo, hi, p_ge0, m


# --------------------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------------------
def realism_models(split):
    """Ordered [(label, config, preds)] for a realism split."""
    rows = []
    for m in common.ZS_STEMS:
        p = common.load_judge_preds(m, split, "zero")
        if p:
            rows.append((m, "Zeroshot", p))
    for m in common.FS_STEMS:
        p = common.load_judge_preds(m, split, "few")
        if p:
            rows.append((m + " (FS)", "Fewshot", p))
    return rows


def build_realism_table(split, id_filter=None):
    meta = common.load_metadata(split)
    dims = common.REALISM_DIMS
    table = {}
    # DRAME (base split reproduces; real-rec clean dir may be empty -> pending)
    drame_dir = {"realism_base": "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall",
                 "realism_real": None}[split if split in ("realism_base", "realism_real") else "realism_base"]
    if split == "realism_base":
        dp = load_dir_preds(drame_dir, dims)
        cells = build_cells(dp, meta, dims, realism_human, id_filter)
        table["DRAME-Eval"] = ("Finetune", cells)
    else:  # realism_real -> clean dir
        clean = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2"
        dp = load_dir_preds(clean, dims)
        if dp:
            cells = build_cells(dp, meta, dims, realism_human, id_filter)
            table["DRAME-Eval"] = ("Finetune", cells)
        else:
            table["DRAME-Eval"] = ("Finetune", None)  # PENDING re-inference
    for label, cfg, preds in realism_models(split):
        cells = build_cells(preds, meta, dims, realism_human, id_filter)
        table[label] = (cfg, cells)
    return meta, dims, table


def fmt(r):
    return " nan " if r != r else f"{r:+.3f}"


def render_realism_block(name, dims, table, note=""):
    lines = [f"### {name}", ""]
    if note:
        lines += [note, ""]
    lines.append("| Config | Model | " + " | ".join(common.REALISM_LABELS[d] for d in dims) + " | **Avg [95% CI]** | n |")
    lines.append("|" + "---|" * (len(dims) + 4))
    for label, (cfg, cells) in table.items():
        if cells is None:
            lines.append(f"| {cfg} | {label} | " + " | ".join(["_pending_"] * len(dims)) + " | _pending re-inference_ | — |")
            continue
        cellstr = []
        nmax = 0
        for d in dims:
            r, lo, hi, n = cell_ci(cells, d)
            cellstr.append(f"{fmt(r)}")
            nmax = max(nmax, n)
        ap, alo, ahi = avg_ci(cells, dims)
        lines.append(f"| {cfg} | {label} | " + " | ".join(cellstr) + f" | **{fmt(ap)} [{fmt(alo)}, {fmt(ahi)}]** | {nmax} |")
    lines.append("")
    return lines


def render_realism_ci_detail(name, dims, table):
    """Per-cell CI detail (each cell as r [lo,hi]) — appendix-grade."""
    lines = [f"### {name} — per-cell 95% CIs", "",
             "| Model | " + " | ".join(common.REALISM_LABELS[d] for d in dims) + " |"]
    lines.append("|" + "---|" * (len(dims) + 1))
    for label, (cfg, cells) in table.items():
        if cells is None:
            lines.append(f"| {label} | " + " | ".join(["_pending_"] * len(dims)) + " |")
            continue
        row = []
        for d in dims:
            r, lo, hi, n = cell_ci(cells, d)
            row.append(f"{fmt(r)} [{fmt(lo)},{fmt(hi)}]")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    out = {"n_boot": N_BOOT, "seed": SEED, "gating": GATING}
    md = ["# E1 — Bootstrap CIs & paired tests (mask gating, 2000 resamples)", "",
          f"Protocol: gating=`{GATING}` (reviewer-primary, release-native); real-rec on "
          f"held-out `test_real_v2` (1,084), verified disjoint from training IDs.", ""]

    # ---------- Tab 8: realism base ----------
    print("Building Tab 8 (realism base) ...")
    meta_b, dims, tab8 = build_realism_table("realism_base")
    md += render_realism_block("Tab 8 — Realism in-distribution base test (Avg with CI)", dims, tab8)
    out["tab8"] = {lab: {"avg": avg_ci(c, dims)[0], "ci": avg_ci(c, dims)[1:]} for lab, (_, c) in tab8.items() if c}

    # ---------- Tab 9: real-rec baselines on test_real_v2 ----------
    print("Building Tab 9 (real-rec, test_real_v2) ...")
    v2 = set(load_v2_ids())
    meta_r, _, tab9 = build_realism_table("realism_real", id_filter=v2)
    md += render_realism_block(
        "Tab 9 — Real-recording test (test_real_v2, 1,084 held-out; Avg with CI)", dims, tab9,
        note="_Clean DRAME-Eval predictions; best baseline = GPT4o-audio (FS)._")
    out["tab9"] = {lab: {"avg": avg_ci(c, dims)[0], "ci": avg_ci(c, dims)[1:]} for lab, (_, c) in tab9.items() if c}

    # ---------- Paired deltas ----------
    print("Paired bootstrap deltas ...")
    md += ["## Paired bootstrap Δ(Avg) — same resample indices, per-dim common ids", "",
           "| Split | Comparison A − B | Δ Avg | 95% CI | p(Δ≤0) | n |", "|---|---|---|---|---|---|"]
    pairs_base = [("DRAME-Eval", "GPT4o-audio (FS)"), ("DRAME-Eval", "Gemini2.5Pro (FS)"),
                  ("DRAME-Eval", "Gemini2.5Pro")]
    out["paired"] = {}
    for A, B in pairs_base:
        if A in tab8 and B in tab8 and tab8[A][1] and tab8[B][1]:
            d, lo, hi, p, ncom = paired_delta_avg(tab8[A][1], tab8[B][1], dims)
            md.append(f"| base | {A} − {B} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {p:.3f} | {ncom} |")
            out["paired"][f"base:{A}-{B}"] = {"delta": d, "ci": [lo, hi], "p_le0": p, "n": ncom}
    # Clean real-recording paired comparisons
    for A, B in [("DRAME-Eval", "GPT4o-audio (FS)"), ("GPT4o-audio (FS)", "Gemini2.5Pro (FS)"), ("GPT4o-audio (FS)", "GPT4o-audio")]:
        if A in tab9 and B in tab9 and tab9[A][1] and tab9[B][1]:
            d, lo, hi, p, ncom = paired_delta_avg(tab9[A][1], tab9[B][1], dims)
            md.append(f"| real-rec | {A} − {B} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {p:.3f} | {ncom} |")
            out["paired"][f"real:{A}-{B}"] = {"delta": d, "ci": [lo, hi], "p_le0": p, "n": ncom}
    if tab9.get("DRAME-Eval", (None, None))[1] is None:
        md.append("| real-rec | DRAME-Eval − GPT4o-audio (FS) | _pending_ | _pending_ | _pending_ | — |")
    md.append("")

    # ---------- Tab 12: real-rec ablation on test_real_v2 ----------
    print("Building Tab 12 (real-rec ablation) ...")
    md += ["## Tab 12 — Real-recording fine-tuning ablation (test_real_v2)", "",
           "| Config | Avg [95% CI] | n | status |", "|---|---|---|---|"]
    out["tab12"] = {}
    for label, (relpath, paper, status) in ABLATION_REAL.items():
        preds = load_dir_preds(relpath, dims)
        if not preds:
            md.append(f"| {label} | _pending re-inference_ | — | {status} |")
            out["tab12"][label] = {"status": "pending"}
            continue
        cells = build_cells(preds, meta_r, dims, realism_human, id_filter=v2)
        ap, alo, ahi = avg_ci(cells, dims)
        nmax = max(cell_ci(cells, d)[3] for d in dims)
        md.append(f"| {label} | {fmt(ap)} [{fmt(alo)}, {fmt(ahi)}] | {nmax} | {status} (paper {paper:+.3f}) |")
        out["tab12"][label] = {"avg": ap, "ci": [alo, ahi], "n": nmax, "status": status}
    md.append("")

    # ---------- Tab 11: realism base ablation (mappable) ----------
    print("Building Tab 11 (realism base ablation) ...")
    md += ["## Tab 11 — Realism base fine-tuning ablation (mask; Base-LoRA base dir unresolved)", "",
           "| Config | Avg [95% CI] | n | paper(impute) |", "|---|---|---|---|"]
    out["tab11"] = {}
    for label, (relpath, paper) in ABLATION_BASE.items():
        preds = load_dir_preds(relpath, dims)
        if not preds:
            md.append(f"| {label} | _dir missing_ | — | {paper} |")
            continue
        cells = build_cells(preds, meta_b, dims, realism_human)
        ap, alo, ahi = avg_ci(cells, dims)
        nmax = max(cell_ci(cells, d)[3] for d in dims)
        md.append(f"| {label} | {fmt(ap)} [{fmt(alo)}, {fmt(ahi)}] | {nmax} | {paper:.3f} |")
        out["tab11"][label] = {"avg": ap, "ci": [alo, ahi], "n": nmax}
    md.append("")

    # ---------- Tab 7: archetype DRAME (baselines flagged) ----------
    print("Building Tab 7 (archetype DRAME) ...")
    meta_a = common.load_metadata("archetype")
    dp_a = load_dir_preds(ARCHE_DRAME_DIR, common.ARCHE_DIMS, is_arche=True)
    cells_a = build_cells(dp_a, meta_a, common.ARCHE_DIMS, arche_human)
    md += ["## Tab 7 — Archetype (DRAME-Eval; baselines in E5/E1c)", "",
           "| Model | " + " | ".join(common.ARCHE_LABELS[d] for d in common.ARCHE_DIMS) + " | Avg [95% CI] | n |",
           "|---|" + "---|" * (len(common.ARCHE_DIMS) + 2)]
    row = [f"{fmt(cell_ci(cells_a, d)[0])} [{fmt(cell_ci(cells_a, d)[1])},{fmt(cell_ci(cells_a, d)[2])}]"
           for d in common.ARCHE_DIMS]
    ap, alo, ahi = avg_ci(cells_a, common.ARCHE_DIMS)
    nmax = max(cell_ci(cells_a, d)[3] for d in common.ARCHE_DIMS)
    md.append(f"| DRAME-Eval | " + " | ".join(row) + f" | **{fmt(ap)} [{fmt(alo)}, {fmt(ahi)}]** | {nmax} |")
    md += ["", "Archetype baselines use the same masked protocol; see E5 and E1c "
           "for the consistent baseline rows and paired comparisons.", ""]
    out["tab7_drame"] = {"avg": ap, "ci": [alo, ahi], "n": nmax}

    # ---------- Coverage + Sec.0 contamination ----------
    print("Coverage + contamination audit ...")
    md += ["## Coverage audit (n scored per model, mask gating)", "",
           "| Split | Model | n (Pitch dim) | n (Emotion Intensity) |", "|---|---|---|---|"]
    for split, tab in [("base", tab8), ("real-rec(v2)", tab9)]:
        for label, (cfg, cells) in tab.items():
            if cells is None:
                md.append(f"| {split} | {label} | pending | pending |"); continue
            np_ = len(cells["pitch_variation"][0]); ne = len(cells["emotion_intensity"][0])
            md.append(f"| {split} | {label} | {np_} | {ne} |")
    md.append("")

    # contamination split on the contaminated headline dir
    train_ids = set(common._norm_id(json.loads(l)["id"]) for l in open(TRAIN_V2))
    contam = load_dir_preds(CONTAM_DIR, dims)
    meta_rr = common.load_metadata("realism_real")
    intrain = {r for r in contam if r in train_ids}
    heldout = {r for r in contam if r not in train_ids}

    def avg_on(ids):
        cells = build_cells(contam, meta_rr, dims, realism_human, id_filter=ids)
        return avg_ci(cells, dims)[0], max(len(cells[d][0]) for d in dims)

    r_in, n_in = avg_on(intrain)
    r_out, n_out = avg_on(heldout)
    base_only = load_dir_preds("results/sft_lora_specific_dimension_checkpoint-10000_testreal", dims)
    cells_bo = build_cells(base_only, meta_rr, dims, realism_human)
    r_bo = avg_ci(cells_bo, dims)[0]
    md += ["## Sec.0 contamination split — published 0.247 headline dir (v1 list, 1,084)", "",
           f"- in-train samples (fyr+pro-mufei): **n={n_in}**, Avg r = **{fmt(r_in)}**",
           f"- held-out samples (qyf+zrr+2):    **n={n_out}**, Avg r = **{fmt(r_out)}**",
           f"- base-only model (no real data), all 2,164: Avg r = **{fmt(r_bo)}**",
           "",
           "The published +0.247 headline averages a memorized in-train half against a "
           "held-out half; on the genuinely held-out samples the same run is ~0.05. This is "
           "why the clean test_real_v2 re-inference (yy+zyx, 540 samples) is required before "
           "the clean results must replace the contaminated headline.", ""]
    out["contamination"] = {"in_train_r": r_in, "n_in": n_in, "held_out_r": r_out,
                            "n_out": n_out, "base_only_r": r_bo}

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e1_bootstrap.json"), "w"), indent=2)
    # append the per-cell CI detail tables at the end (appendix-grade)
    md += render_realism_ci_detail("Tab 8 detail", dims, tab8)
    md += render_realism_ci_detail("Tab 9 detail (test_real_v2)", dims, tab9)
    open(os.path.join(common.OUT_DIR, "e1_bootstrap.md"), "w").write("\n".join(md))
    print(f"\nwrote {os.path.join(common.OUT_DIR, 'e1_bootstrap.md')}")
    print(f"wrote {os.path.join(common.OUT_DIR, 'e1_bootstrap.json')}")

    # console highlights
    print("\n--- HIGHLIGHTS ---")
    for lab in ["DRAME-Eval", "Gemini2.5Pro (FS)", "GPT4o-audio (FS)", "Gemini2.5Pro"]:
        if lab in tab8 and tab8[lab][1]:
            p, lo, hi = avg_ci(tab8[lab][1], dims)
            print(f"  base    {lab:20s} Avg {fmt(p)} [{fmt(lo)}, {fmt(hi)}]")
    for lab in ["GPT4o-audio (FS)", "Gemini2.5Pro (FS)", "Qwen2.5Omni", "GPT4o-audio"]:
        if lab in tab9 and tab9[lab][1]:
            p, lo, hi = avg_ci(tab9[lab][1], dims)
            print(f"  realrec {lab:20s} Avg {fmt(p)} [{fmt(lo)}, {fmt(hi)}]")
    print(f"  contamination: in-train {fmt(r_in)} (n={n_in}) vs held-out {fmt(r_out)} (n={n_out}); base-only {fmt(r_bo)}")


if __name__ == "__main__":
    main()
