"""
E3 — Per-language (EN / ZH) breakdowns for the main tables (reviewer NhWo C1 / gEEq R4).

The paper pools English + Mandarin in Tabs. 7-9; both reviewers ask for per-language
tables plus a macro-average, and flag that pooling can hide within-language weakness.

This produces:
  * Tab 8 (realism base): EN (1,000) / ZH (1,000) per-dim points + Avg[CI] + macro-avg, all models
  * Tab 7 (archetype):    EN (800) / ZH (699) for DRAME-Eval (baselines flagged, held back)
  * Confirms the real-recording set is Mandarin-only (2,164 zh) -> Tab 9 caption fix

Reuses the E1 bootstrap machinery (mask gating). Run from analysis/:
  PYTHONPATH=. python3 exp_e3_perlang.py
Writes analysis/out/e3_perlang.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from exp_e1_bootstrap import (
    N_BOOT, SEED, ARCHE_DRAME_DIR, load_dir_preds, realism_human, arche_human,
    realism_models, build_cells, avg_ci, _union_aligned, _avg_from_aligned, fmt,
)

DRAME_BASE_DIR = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


def lang_ids(meta, lang):
    return {rid for rid, r in meta.items() if r.get("language") == lang}


def macro_avg_ci(cells_by_lang, dims, n=N_BOOT, seed=SEED):
    """Macro-average = mean over languages of each language's Avg-over-dims Pearson.

    Bootstrap: resample each language's union ids independently (shared seed stream),
    recompute each language's avg, then average across languages.
    """
    langs = list(cells_by_lang)
    points = [np.nanmean([common.pearson(*cells_by_lang[l][d][1:]) for d in dims]) for l in langs]
    point = float(np.nanmean(points))
    aligned = {l: _union_aligned(cells_by_lang[l], dims) for l in langs}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        per_lang = []
        for l in langs:
            m, per_dim = aligned[l]
            if m == 0:
                per_lang.append(np.nan); continue
            per_lang.append(_avg_from_aligned(per_dim, rng.integers(0, m, m)))
        stats.append(np.nanmean(per_lang))
    stats = np.array(stats); stats = stats[np.isfinite(stats)]
    lo, hi = (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))) if len(stats) else (np.nan, np.nan)
    return point, lo, hi


def per_lang_row(preds, meta, dims, human_fn, langs=("en", "zh")):
    """Return {lang: cells} and per-lang (avg,ci) + macro (avg,ci) + per-dim points."""
    cells_by_lang = {l: build_cells(preds, meta, dims, human_fn, id_filter=lang_ids(meta, l)) for l in langs}
    res = {}
    for l in langs:
        ap, lo, hi = avg_ci(cells_by_lang[l], dims)
        perdim = {d: common.pearson(*cells_by_lang[l][d][1:]) for d in dims}
        n = max(len(cells_by_lang[l][d][0]) for d in dims)
        res[l] = {"avg": ap, "ci": [lo, hi], "perdim": perdim, "n": n}
    mp, mlo, mhi = macro_avg_ci(cells_by_lang, dims)
    res["macro"] = {"avg": mp, "ci": [mlo, mhi]}
    return res


def render_realism(name, dims, rows, langs=("en", "zh")):
    lines = [f"### {name}", ""]
    for l in langs:
        lines += [f"**{l.upper()}**", "",
                  "| Config | Model | " + " | ".join(common.REALISM_LABELS[d] for d in dims) + " | Avg [95% CI] | n |",
                  "|" + "---|" * (len(dims) + 4)]
        for label, (cfg, res) in rows.items():
            if res is None:
                lines.append(f"| {cfg} | {label} | " + " | ".join(["_pending_"] * len(dims)) + " | _pending_ | — |")
                continue
            r = res[l]
            perdim = " | ".join(fmt(r["perdim"][d]) for d in dims)
            lines.append(f"| {cfg} | {label} | {perdim} | {fmt(r['avg'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] | {r['n']} |")
        lines.append("")
    # macro-average summary
    lines += ["**Macro-average (mean of EN, ZH)**", "",
              "| Config | Model | EN Avg | ZH Avg | Macro Avg [95% CI] | pooled (E1) |", "|---|---|---|---|---|---|"]
    for label, (cfg, res) in rows.items():
        if res is None:
            lines.append(f"| {cfg} | {label} | _pending_ | _pending_ | _pending_ | _pending_ |"); continue
        m = res["macro"]
        lines.append(f"| {cfg} | {label} | {fmt(res['en']['avg'])} | {fmt(res['zh']['avg'])} | "
                     f"{fmt(m['avg'])} [{fmt(m['ci'][0])}, {fmt(m['ci'][1])}] | — |")
    lines.append("")
    return lines


def main():
    out = {"n_boot": N_BOOT, "seed": SEED, "gating": "mask"}
    md = ["# E3 — Per-language (EN/ZH) breakdowns + macro-average (mask gating)", "",
          "Pooled numbers are in E1. Real-recording set is **Mandarin-only (2,164 zh)** — "
          "Tab 9's 'English+Mandarin pooled' caption must be corrected to Mandarin-only, and "
          "EN real-recording collection noted as future work.", ""]

    # ---- Tab 8 realism base ----
    print("E3 Tab 8 (realism base) per-language ...")
    meta_b = common.load_metadata("realism_base")
    dims = common.REALISM_DIMS
    rows8 = {}
    dp = load_dir_preds(DRAME_BASE_DIR, dims)
    rows8["DRAME-Eval"] = ("Finetune", per_lang_row(dp, meta_b, dims, realism_human))
    for label, cfg, preds in realism_models("realism_base"):
        rows8[label] = (cfg, per_lang_row(preds, meta_b, dims, realism_human))
    md += render_realism("Tab 8 — Realism base test, per-language", dims, rows8)
    out["tab8"] = {lab: r for lab, (_, r) in rows8.items()}

    # ---- Tab 7 archetype (DRAME only) ----
    print("E3 Tab 7 (archetype DRAME) per-language ...")
    meta_a = common.load_metadata("archetype")
    adims = common.ARCHE_DIMS
    dp_a = load_dir_preds(ARCHE_DRAME_DIR, adims, is_arche=True)
    res_a = per_lang_row(dp_a, meta_a, adims, arche_human)
    md += ["### Tab 7 — Archetype, per-language (DRAME-Eval; baselines flagged/held)", ""]
    for l in ("en", "zh"):
        r = res_a[l]
        md += [f"**{l.upper()}** — " + ", ".join(f"{common.ARCHE_LABELS[d]} {fmt(r['perdim'][d])}" for d in adims)
               + f"  →  Avg {fmt(r['avg'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] (n={r['n']})"]
    md += ["", f"**Macro-avg** {fmt(res_a['macro']['avg'])} [{fmt(res_a['macro']['ci'][0])}, {fmt(res_a['macro']['ci'][1])}]",
           "", "> Archetype baseline masked rows and paired comparisons are reported by E5 and E1c.", ""]
    out["tab7_drame"] = res_a

    # ---- real-rec language confirmation ----
    meta_r = common.load_metadata("realism_real")
    langs_r = collections.Counter(r.get("language") for r in meta_r.values())
    md += ["### Tab 9 — real-recording language composition", "",
           f"- languages: {dict(langs_r)} → **Mandarin-only**. No EN real-recording data exists anywhere in the release.",
           "- Action: fix Tab 9 caption ('English+Mandarin pooled' → 'Mandarin only'); add EN real-rec to limitations/future work.", ""]
    out["realrec_langs"] = dict(langs_r)

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e3_perlang.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e3_perlang.md"), "w").write("\n".join(md))
    print(f"wrote {os.path.join(common.OUT_DIR, 'e3_perlang.md')}")

    # console highlights: the reviewer's worry is "pooling hides within-language weakness"
    print("\n--- E3 HIGHLIGHTS (Avg by language) ---")
    for lab, (_, r) in rows8.items():
        print(f"  base {lab:20s} EN {fmt(r['en']['avg'])}  ZH {fmt(r['zh']['avg'])}  macro {fmt(r['macro']['avg'])}")
    print(f"  arche DRAME-Eval        EN {fmt(res_a['en']['avg'])}  ZH {fmt(res_a['zh']['avg'])}  macro {fmt(res_a['macro']['avg'])}")


if __name__ == "__main__":
    main()
