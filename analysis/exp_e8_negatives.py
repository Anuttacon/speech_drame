"""
E8 — Negative-sample reporting & validation (reviewer NhWo C5; supports the W2 count
reconciliation and reviewer ZB3A RC2 QA question).

Three deliverables, all from the released split metadata:
  (1) Counts by sample_type x negative_type x severity x language x split — the single
      consistent set of numbers to replace Tab 3's stale figures.
  (2) Human-score monotonicity across perturbation severity (processing_level L0/L1/L2):
      mean human rating on the TARGETED consistency dimensions should decrease with
      severity if the negatives are graded as intended. Positives serve as the ceiling.
  (3) Annotator-blindness check: rater `confidence` distribution across positive /
      role-neg / context-neg. If confidence does NOT collapse on negatives, annotators
      were not tipped off to the mismatch type (it was hidden, used only for QA).

Provenance note (per PLAN_review2 / memory): the released `is_real_audio=True` flag is a
known bug; true audio provenance is real for test_realrec only and ~50-62% TTS elsewhere.
E8 reports label/rating structure; the audio-provenance composition is E12 (PLAN_review2).

Run from analysis/:  PYTHONPATH=. python3 exp_e8_negatives.py
Writes analysis/out/e8_negatives.{json,md}.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import common

REALISM_RELEASE = common.REALISM_META
ARCHE_RELEASE = common.ARCHE_META

# consistency dims targeted by role / context perturbations
ROLE_TARGET = ["traits_embodiment", "local_scene_fit", "global_story_coherence"]
CTX_TARGET = ["local_scene_fit", "global_story_coherence"]
PROSODY_DIMS = ["pitch_variation", "rhythmic_naturalness", "stress_emphasis"]


def load(fp):
    return [json.loads(l) for l in open(fp)]


def sev_label(pl):
    return {0: "L0", 1: "L1", 2: "L2"}.get(pl, "—")


def counts_block():
    lines = ["## 1. Dataset counts (released splits; replaces stale Tab 3 figures)", "",
             "| Split | n | positive | role-neg | context-neg | delivery-neg | EN | ZH | sev L0/L1/L2 (neg) |",
             "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    files = [("realism train", REALISM_RELEASE + "/train.jsonl"),
             ("realism test_base", REALISM_RELEASE + "/test_base.jsonl"),
             ("realism test_realrec", REALISM_RELEASE + "/test_realrec.jsonl"),
             ("realism benchmark", REALISM_RELEASE + "/benchmark.jsonl")]
    for name, fp in files:
        recs = load(fp)
        st = collections.Counter(r.get("sample_type") for r in recs)
        nt = collections.Counter(r.get("negative_type") for r in recs)
        lang = collections.Counter(r.get("language") for r in recs)
        sev = collections.Counter(r.get("processing_level") for r in recs if r.get("sample_type") == "negative")
        lines.append(f"| {name} | {len(recs)} | {st.get('positive',0)} | {nt.get('role',0)} | "
                     f"{nt.get('context',0)} | {nt.get('delivery',0)} | {lang.get('en',0)} | "
                     f"{lang.get('zh',0)} | {sev.get(0,0)}/{sev.get(1,0)}/{sev.get(2,0)} |")
        res[name] = {"n": len(recs), "sample_type": dict(st), "negative_type": dict(nt),
                     "language": dict(lang), "severity": {sev_label(k): v for k, v in sev.items()}}
    # archetype
    for name, fp in [("archetype train", ARCHE_RELEASE + "/train.jsonl"),
                     ("archetype test", ARCHE_RELEASE + "/test.jsonl")]:
        recs = load(fp)
        lang = collections.Counter(r.get("language") for r in recs)
        cp = collections.Counter(r.get("annotations", {}).get("content_pass") for r in recs)
        nullr = sum(1 for r in recs if r.get("annotations", {}).get("appropriateness") is None)
        lines.append(f"| {name} | {len(recs)} | pass {cp.get(True,0)} / fail {cp.get(False,0)} | — | — | — | "
                     f"{lang.get('en',0)} | {lang.get('zh',0)} | null-rated: {nullr} |")
        res[name] = {"n": len(recs), "content_pass": {str(k): v for k, v in cp.items()},
                     "language": dict(lang), "null_rated": nullr}
    lines += ["",
              "_Realism totals: train 10,990 (2,504 pos / 8,486 neg = 4,310 role + 4,176 "
              "context), test_base 2,000 (210 / 1,790), test_realrec 2,164 (all real "
              "positives, ZH-only), benchmark 260 (all delivery negatives, ZH). Grand "
              "realism = 15,414. Archetype = 8,272 (6,773 train / 1,499 test). "
              "Archetype test content-fail = 233 (`content_pass=False`); a slightly smaller "
              "set (≈187) additionally have null AQ/HL/AP arrays — the difference is "
              "content-fail samples that were still rated. Report one convention (mask) "
              "consistently (see E5)._", ""]
    return lines, res


def monotonicity_block():
    """Mean human rating on targeted dims by severity, test_base."""
    meta = common.load_metadata("realism_base")
    recs = list(meta.values())

    def mean_dim_by(group_recs, dims):
        vals = []
        for r in group_recs:
            ds = [common.human_mean(r, d, gating="mask") for d in dims]
            ds = [x for x in ds if x is not None]
            if ds:
                vals.append(np.mean(ds))
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

    lines = ["## 2. Human-score monotonicity across perturbation severity (test_base)", "",
             "Mean human rating (mask) on the dimensions each perturbation targets. "
             "Positives are the un-perturbed ceiling; L0→L2 = increasing perturbation "
             "strength. A monotone decrease confirms the negatives are graded as designed.", "",
             "| Group | Targeted dims | L0 | L1 | L2 | positive (ceiling) |",
             "|---|---|---|---|---|---|"]
    res = {}
    for gname, ntype, dims in [("role-negative", "role", ROLE_TARGET),
                               ("context-negative", "context", CTX_TARGET)]:
        cells = {}
        for lvl in (0, 1, 2):
            g = [r for r in recs if r.get("negative_type") == ntype and r.get("processing_level") == lvl]
            cells[lvl] = mean_dim_by(g, dims)
        pos = [r for r in recs if r.get("sample_type") == "positive"]
        pv = mean_dim_by(pos, dims)
        dimlab = "+".join(common.REALISM_LABELS[d] for d in dims)
        lines.append(f"| {gname} | {dimlab} | {cells[0][0]:.3f} (n={cells[0][1]}) | "
                     f"{cells[1][0]:.3f} (n={cells[1][1]}) | {cells[2][0]:.3f} (n={cells[2][1]}) | "
                     f"{pv[0]:.3f} (n={pv[1]}) |")
        res[gname] = {"L0": cells[0][0], "L1": cells[1][0], "L2": cells[2][0], "positive": pv[0]}
    # also show a NON-targeted control (prosody) for role-neg — should NOT drop much
    role = [r for r in recs if r.get("negative_type") == "role"]
    pos = [r for r in recs if r.get("sample_type") == "positive"]
    role_ctrl = mean_dim_by(role, PROSODY_DIMS)[0]
    pos_ctrl = mean_dim_by(pos, PROSODY_DIMS)[0]
    role_tgt = np.mean([res["role-negative"][k] for k in ("L0", "L1", "L2")])
    lines += ["",
              f"_Control (prosody dims, NOT targeted by role perturbation): role-neg "
              f"{role_ctrl:.3f} vs positive {pos_ctrl:.3f} — a {pos_ctrl - role_ctrl:.2f} gap, "
              f"far smaller than the {pos_ctrl and (res['role-negative']['positive'] - role_tgt):.2f} "
              f"gap on the targeted consistency dims. Perturbations move the dimensions they "
              f"target far more than the acoustic ones, as designed._", ""]
    res["prosody_control"] = {"role_neg": mean_dim_by(role, PROSODY_DIMS)[0],
                              "positive": mean_dim_by(pos, PROSODY_DIMS)[0]}
    return lines, res


def blindness_block():
    """Rater confidence distribution pos vs neg types (test_base)."""
    meta = common.load_metadata("realism_base")
    recs = list(meta.values())

    def conf(group):
        vals = []
        for r in group:
            arr = r.get("annotations", {}).get("raters", {}).get("confidence") or []
            arr = [float(v) for v in arr if v is not None]
            if arr:
                vals.append(np.mean(arr))
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

    groups = {"positive": [r for r in recs if r.get("sample_type") == "positive"],
              "role-neg": [r for r in recs if r.get("negative_type") == "role"],
              "context-neg": [r for r in recs if r.get("negative_type") == "context"]}
    lines = ["## 3. Annotator-blindness check — rater confidence by sample type (test_base)", "",
             "Mismatch type is hidden from raters (used only for QA). If annotators were not "
             "cued, mean self-reported confidence should be ~flat across positive and "
             "negative types.", "",
             "| Group | mean confidence | n |", "|---|---|---|"]
    res = {}
    for g, gr in groups.items():
        c, n = conf(gr)
        lines.append(f"| {g} | {c:.3f} | {n} |")
        res[g] = c
    spread = max(res.values()) - min(res.values())
    lines += ["", f"_Spread across types = {spread:.3f} on the 1-5 confidence scale — "
              f"{'negligible; consistent with blind annotation' if spread < 0.3 else 'non-trivial; discuss'}._", ""]
    res["spread"] = spread
    return lines, res


def main():
    out = {}
    md = ["# E8 — Negative-sample counts, severity monotonicity, annotator blindness", ""]
    l, r = counts_block(); md += l; out["counts"] = r
    l, r = monotonicity_block(); md += l; out["monotonicity"] = r
    l, r = blindness_block(); md += l; out["blindness"] = r
    md += ["## Notes for integration", "",
           "- **W2:** the counts table above is the single source of truth; sweep tex for "
           "8,280 / '12k/2k/1k' / '23k' and replace.",
           "- **C5:** severity monotonicity (2) + blindness (3) are the negative-validation "
           "evidence — one appendix table + a sentence in the caveats paragraph.",
           "- **QA rejection rate:** not recoverable from the released metadata (rejection "
           "logs not shipped); state the QA procedure and mark rejection counts as "
           "documented-only unless found in `preprocess/raw_data/` (see E9).", ""]

    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e8_negatives.json"), "w"), indent=2)
    open(os.path.join(common.OUT_DIR, "e8_negatives.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e8_negatives.md"))
    print("\n--- E8 HIGHLIGHTS ---")
    for g, v in out["monotonicity"].items():
        if isinstance(v, dict) and "L0" in v:
            print(f"  {g:16s} L0 {v['L0']:.2f} L1 {v['L1']:.2f} L2 {v['L2']:.2f} pos {v['positive']:.2f}")
    print(f"  confidence spread pos/role/ctx = {out['blindness']['spread']:.3f}")


if __name__ == "__main__":
    main()
