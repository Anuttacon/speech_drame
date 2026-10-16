"""
E15 — Archetype per-SFnd-system correlation breakdown (reviewer gEEq; concession package for
the archetype track). The archetype split is role×scene-disjoint train/test but SYSTEM-shared
(the same 8 SFnd synthesis systems appear in train and test), so a reviewer may worry the
0.629 headline reflects between-system contrast rather than within-system acting judgment.

We report, per system (avg over AQ/HL/AP, mask gating):
  - DRAME-Eval + the best zero/few-shot judges' correlation within each system;
  - the pooled vs macro(mean-over-systems) correlation — a large pooled>macro gap would mean
    the headline is inflated by between-system mean shift (the archetype analogue of E3's
    between-language pooling inflation).

Run from analysis/:  PYTHONPATH=. python3 exp_e15_archetype_persystem.py
Writes analysis/out/e15_archetype_persystem.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common
from exp_e1_bootstrap import ARCHE_DRAME_DIR, load_dir_preds, arche_human, build_cells, fmt


def system_of(rec):
    return rec.get("sfnd", {}).get("system")


def per_system_avg(preds, meta, dims, systems):
    """{system: avg-over-dims Pearson}, mask gating."""
    res = {}
    for sysname in systems:
        ids = {rid for rid, r in meta.items() if system_of(r) == sysname}
        cells = build_cells(preds, meta, dims, arche_human, id_filter=ids)
        rs = [common.pearson(*cells[d][1:]) for d in dims]
        n = max(len(cells[d][0]) for d in dims)
        res[sysname] = (float(np.nanmean(rs)), n)
    return res


def pooled_avg(preds, meta, dims):
    cells = build_cells(preds, meta, dims, arche_human)
    return float(np.nanmean([common.pearson(*cells[d][1:]) for d in dims]))


def main():
    meta = common.load_metadata("archetype")
    dims = common.ARCHE_DIMS
    systems = [s for s, _ in collections.Counter(system_of(r) for r in meta.values()).most_common()]
    out = {"systems": systems}

    models = [("DRAME-Eval", load_dir_preds(ARCHE_DRAME_DIR, dims, is_arche=True))]
    for m in ("Gemini2.5Pro", "GPT4o-audio"):
        p = common.load_judge_preds_arche(m, "zero")
        if p:
            models.append((m + " (ZS)", p))

    md = ["# E15 — Archetype per-SFnd-system correlation (mask gating)", "",
          "Avg over AQ/HL/AP per system. The split is role×scene-disjoint but system-shared; "
          "this shows whether the correlation holds within each synthesis system or is a "
          "between-system contrast.", "",
          "| Model | " + " | ".join(systems) + " | **pooled** | macro (mean-of-sys) |",
          "|---|" + "---|" * (len(systems) + 2)]
    out["per_system"] = {}
    for label, preds in models:
        ps = per_system_avg(preds, meta, dims, systems)
        pooled = pooled_avg(preds, meta, dims)
        macro = float(np.nanmean([ps[s][0] for s in systems if ps[s][0] == ps[s][0]]))
        cells = " | ".join(f"{fmt(ps[s][0])} (n={ps[s][1]})" for s in systems)
        md.append(f"| {label} | {cells} | **{fmt(pooled)}** | {fmt(macro)} |")
        out["per_system"][label] = {"per_system": {s: ps[s][0] for s in systems},
                                    "pooled": pooled, "macro": macro}
    md += ["",
           "_Read the pooled vs macro gap: if pooled ≫ macro, the headline is inflated by "
           "between-system mean shift (systems differ in overall quality and both humans and "
           "the model track that). DRAME-Eval's within-system correlations show whether it "
           "discriminates acting quality *within* a fixed synthesis system — the harder, "
           "more meaningful task._", "",
           "## Integration (concession)", "",
           "- State in Limitations + Tab 7 caption: archetype train/test share the 8 SFnd "
           "systems (disjoint only by role×scene). Report per-system correlations so the "
           "claim is 'consistent within each system', not just pooled.",
           "- If a leave-one-system-out retrain is approved (D2), that is the definitive "
           "answer; otherwise narrow the claim to system-shared generalization.", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e15_archetype_persystem.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e15_archetype_persystem.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e15_archetype_persystem.md"))
    print("\n--- E15 HIGHLIGHTS ---")
    for label, d in out["per_system"].items():
        vals = [f"{s.split('-')[0][:8]}:{fmt(v)}" for s, v in d["per_system"].items()]
        print(f"  {label:18s} pooled {fmt(d['pooled'])} macro {fmt(d['macro'])}")


if __name__ == "__main__":
    main()
