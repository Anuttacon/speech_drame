"""
Experiment 3 — System-level ranking (archetype).

A judge's practical job is to *rank generation systems* the way humans do; sample-level Pearson
(Tables 2-4) undersells this. On the archetype test set (8 SFnd systems, 100-200 samples each)
we aggregate each model's predictions to a per-system mean per dimension and correlate those
system means against the human per-system means.

For each model and dimension we report system-level Pearson / Spearman / Kendall-tau across the
8 systems, plus the unweighted average over the three archetype dimensions. With only 8 systems
the rank correlations (Spearman/Kendall) are the more meaningful summary and CIs are wide; both
are reported honestly.

Outputs: analysis/out/exp3_system_ranking.json and analysis/out/exp3.md
"""
import json
import os

import numpy as np

import common as C

DIMS = C.ARCHE_DIMS  # audio_quality, human_likeness, appropriateness
JUDGES_ZERO = ["Gemini2.5Pro", "GPT4o-audio", "KimiAudio", "Qwen2Audio", "Qwen2.5Omni"]
JUDGES_FEW = ["Gemini2.5Pro", "GPT4o-audio"]


def per_system_means(meta, score_of, dim):
    """Return {system: mean score} where score_of(rid)->value or None, over the metadata."""
    agg = {}
    for rid, rec in meta.items():
        v = score_of(rid)
        if v is None:
            continue
        sysname = rec.get("sfnd", {}).get("system")
        if not sysname:
            continue
        agg.setdefault(sysname, []).append(v)
    return {s: float(np.mean(vs)) for s, vs in agg.items()}


def systems_corr(human_by_sys, model_by_sys):
    """Align on shared systems, return (pearson, spearman, kendall, n_systems)."""
    systems = sorted(set(human_by_sys) & set(model_by_sys))
    h = [human_by_sys[s] for s in systems]
    m = [model_by_sys[s] for s in systems]
    return C.pearson(h, m), C.spearman(h, m), C.kendall(h, m), len(systems)


def run():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    meta = C.load_metadata("archetype")

    # human per-system means per dimension (gating mask)
    human = {dim: per_system_means(meta, lambda rid, d=dim: C.human_mean(meta[rid], d), dim) for dim in DIMS}

    # model registry
    models = []
    for m in JUDGES_ZERO:
        models.append(("Zeroshot", m, C.load_judge_preds_arche(m, "zero")))
    for m in JUDGES_FEW:
        p = C.load_judge_preds_arche(m, "few")
        if p:
            models.append(("Fewshot", m, p))
    models.append(("Finetune", "DRAME-Eval", C.load_drame_preds("archetype")))

    results = {"human_per_system": human, "models": {}}
    md = ["## Experiment 3 — System-level ranking (archetype)\n"]
    n_sys = len(set().union(*[set(human[d]) for d in DIMS]))
    md.append(
        f"Per-system means across **{n_sys} SFnd systems** (archetype test). Each cell is the "
        "system-level correlation between a model's per-system mean and the human per-system mean. "
        "_P/S/K_ = Pearson / Spearman / Kendall-tau; _Avg_ columns average the three archetype "
        "dimensions. With 8 systems the rank correlations are the meaningful summary. For reference, "
        "the paper's *sample-level* archetype averages are 0.480 (Gemini zero-shot) and 0.629 "
        "(DRAME-Eval).\n")

    # reference: human per-system table
    systems = sorted(set().union(*[set(human[d]) for d in DIMS]))
    md.append("\n**Human mean score per system (sanity vs Appendix table):**\n")
    md.append("| System | " + " | ".join(C.ARCHE_LABELS[d] for d in DIMS) + " |")
    md.append("|---|" + "---|" * len(DIMS))
    for s in systems:
        md.append(f"| {s} | " + " | ".join(f"{human[d].get(s, float('nan')):.3f}" for d in DIMS) + " |")

    md.append("\n**System-level model-vs-human ranking correlation:**\n")
    md.append("| Config | Model | " + " | ".join(f"{C.ARCHE_LABELS[d]} (P/S/K)" for d in DIMS) +
              " | Avg-P | Avg-S | Avg-K |")
    md.append("|---|---|" + "---|" * (len(DIMS) + 3))

    for cfg, name, preds in models:
        if preds is None:
            continue
        row = {}
        cells = []
        ps, ss, ks = [], [], []
        for dim in DIMS:
            mbs = per_system_means(meta, lambda rid, d=dim: preds.get(rid, {}).get(d), dim)
            p, s, k, ns = systems_corr(human[dim], mbs)
            row[dim] = {"pearson": p, "spearman": s, "kendall": k, "n_systems": ns}
            cells.append(f"{p:.2f}/{s:.2f}/{k:.2f}")
            ps.append(p); ss.append(s); ks.append(k)
        md.append(f"| {cfg} | {name} | " + " | ".join(cells) +
                  f" | {np.nanmean(ps):.3f} | {np.nanmean(ss):.3f} | {np.nanmean(ks):.3f} |")
        results["models"][f"{cfg}:{name}"] = row

    with open(os.path.join(C.OUT_DIR, "exp3_system_ranking.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(C.OUT_DIR, "exp3.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n[written] analysis/out/exp3_system_ranking.json, exp3.md")


if __name__ == "__main__":
    run()
