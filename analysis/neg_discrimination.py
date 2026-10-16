"""
Experiment 2 — Positive vs negative discrimination.

The contrastive negatives are billed as a first-class contribution but are never used to
*measure* a model in the paper. Here we close that gap on the in-distribution realism set
(test_base = 210 positives + 832 role + 958 context negatives), all real audio with a single
component perturbed.

2a  Human validation (label-only): on the dimension(s) the perturbation targets, do human
    raters score negatives below matched positives? Reported as mean(pos), mean(neg), gap,
    rank-biserial effect size, and a Mann-Whitney p-value.

2b  Model discrimination: for every judge (zero-shot, few-shot) and DRAME-Eval, can the model's
    score on the targeted dimension separate positives from negatives?
      * AUROC (higher model score => 'positive'); 0.5 = chance, <0.5 = wrong direction.
      * score-gap mean(pos) - mean(neg) with 95% bootstrap CI.
    A model that truly attends to the perturbed cue should show AUROC well above 0.5 and a gap
    comparable to the human gap; a transcript-pattern-matcher stays near 0.5.

Targeted dimensions per negative type:
    context (C_scene mismatch) -> local_scene_fit, global_story_coherence
    role    (C_profile+scene)  -> voice_identity_matching, traits_embodiment

Delivery negatives are not covered for the judges (no saved predictions); see exp2_delivery.

Outputs: analysis/out/exp2_neg_discrimination.json and analysis/out/exp2.md
"""
import json
import os

import numpy as np
import scipy.stats

import common as C

SPLIT = "realism_base"
TARGETS = {
    "context": ["local_scene_fit", "global_story_coherence"],
    "role": ["voice_identity_matching", "traits_embodiment"],
}
JUDGES_ZERO = ["Gemini2.5Pro", "GPT4o-audio", "KimiAudio", "Qwen2Audio", "Qwen2.5Omni"]
JUDGES_FEW = ["Gemini2.5Pro", "GPT4o-audio"]


def rank_biserial(pos, neg):
    """Effect size from Mann-Whitney U: 2*AUC-1 oriented as (pos>neg)."""
    a = C.auroc(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    return 2 * a - 1


def mw_p(pos, neg):
    if len(pos) < 1 or len(neg) < 1:
        return float("nan")
    try:
        return float(scipy.stats.mannwhitneyu(pos, neg, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def gap_ci(pos, neg, seed=0, n=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)

    def stat(rng):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        return p.mean() - q.mean()

    return C.bootstrap_ci(stat, n=n, seed=seed)


def auroc_ci(scores, labels, seed=0, n=2000):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    m = len(scores)

    def stat(rng):
        idx = rng.integers(0, m, m)
        return C.auroc(scores[idx], labels[idx])

    return C.bootstrap_ci(stat, n=n, seed=seed)


def model_targeted_scores(preds, meta, ids, dims):
    """Mean over targeted dims of a model's score, for each id present; returns {id: score}."""
    out = {}
    for rid in ids:
        pr = preds.get(rid)
        if not pr:
            continue
        vals = [pr[d] for d in dims if d in pr]
        if vals:
            out[rid] = float(np.mean(vals))
    return out


def run():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    meta = C.load_metadata(SPLIT)
    by_type = {"positive": [], "role": [], "context": []}
    for rid, r in meta.items():
        if r.get("sample_type") == "positive":
            by_type["positive"].append(rid)
        else:
            nt = r.get("negative_type")
            if nt in by_type:
                by_type[nt].append(rid)
    pos_ids = by_type["positive"]

    # load all model predictions on this split once
    model_preds = {}
    for m in JUDGES_ZERO:
        model_preds[("zero", m)] = C.load_judge_preds(m, SPLIT, shot="zero")
    for m in JUDGES_FEW:
        p = C.load_judge_preds(m, SPLIT, shot="few")
        if p:
            model_preds[("few", m)] = p
    model_preds[("ft", "DRAME-Eval")] = C.load_drame_preds(SPLIT)

    results = {}
    md = ["## Experiment 2 — Positive vs negative discrimination\n"]
    md.append(
        f"In-distribution realism set (`test_base`): **{len(pos_ids)} positives, "
        f"{len(by_type['role'])} role negatives, {len(by_type['context'])} context negatives** "
        "(all real audio, one component perturbed). Each negative type is compared against the "
        "positives on the dimension(s) its perturbation targets.\n")

    for ntype, dims in TARGETS.items():
        neg_ids = by_type[ntype]
        dim_label = " + ".join(C.REALISM_LABELS[d] for d in dims)
        md.append(f"\n### {ntype}-mismatch negatives  (target: {dim_label};  "
                  f"{len(pos_ids)} pos vs {len(neg_ids)} neg)\n")

        # ---- 2a human validation ----
        hp = np.array([v for i in pos_ids for d in dims if (v := C.human_mean(meta[i], d)) is not None])
        hn = np.array([v for i in neg_ids for d in dims if (v := C.human_mean(meta[i], d)) is not None])
        gap = hp.mean() - hn.mean()
        glo, ghi = gap_ci(hp, hn)
        md.append("**2a Human validation** (targeted dims, pooled):")
        md.append(f"- mean(pos) = {hp.mean():.3f}, mean(neg) = {hn.mean():.3f}, "
                  f"gap = {C.fmt_ci(gap, glo, ghi)}, rank-biserial = {rank_biserial(hp, hn):.3f}, "
                  f"Mann-Whitney p = {mw_p(hp, hn):.1e}\n")
        results.setdefault(ntype, {})["human"] = {
            "mean_pos": float(hp.mean()), "mean_neg": float(hn.mean()),
            "gap": float(gap), "gap_ci": [glo, ghi], "n_pos": len(pos_ids), "n_neg": len(neg_ids)}

        # ---- 2b model discrimination ----
        md.append("**2b Model discrimination** (targeted-dim score separating pos vs neg):")
        md.append("| Config | Model | AUROC | score-gap (pos−neg) | n(pos/neg) |")
        md.append("|---|---|---|---|---|")
        results[ntype]["models"] = {}
        for (cfg, m), preds in model_preds.items():
            if preds is None:
                continue
            ms = model_targeted_scores(preds, meta, pos_ids + neg_ids, dims)
            ps = [ms[i] for i in pos_ids if i in ms]
            ns = [ms[i] for i in neg_ids if i in ms]
            if len(ps) < 2 or len(ns) < 2:
                continue
            scores = np.array(ps + ns)
            labels = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
            au = C.auroc(scores, labels)
            alo, ahi = auroc_ci(scores, labels)
            g = np.mean(ps) - np.mean(ns)
            glo2, ghi2 = gap_ci(ps, ns)
            cfgname = {"zero": "Zeroshot", "few": "Fewshot", "ft": "Finetune"}[cfg]
            md.append(f"| {cfgname} | {m} | {C.fmt_ci(au, alo, ahi)} | "
                      f"{C.fmt_ci(g, glo2, ghi2)} | {len(ps)}/{len(ns)} |")
            results[ntype]["models"][f"{cfg}:{m}"] = {
                "auroc": au, "auroc_ci": [alo, ahi], "gap": float(g), "gap_ci": [glo2, ghi2],
                "n_pos": len(ps), "n_neg": len(ns)}

    md.append("\n### delivery-mismatch negatives (scope note)\n")
    md.append(
        "Per the agreed scope, the delivery branch is not reported here: (i) saved predictions on "
        "the 260 delivery negatives exist only for DRAME-Eval, not the zero-/few-shot judges, and "
        "(ii) the released delivery set (`benchmark.jsonl`) is 260 delivery negatives with **no "
        "in-set positive reference**, so a controlled positive-vs-negative AUROC would conflate "
        "synthetic-vs-real with the delivery perturbation itself. DRAME-Eval's behavior on "
        "TTS-vs-human delivery is instead visible in the per-system benchmark scoring "
        "(`benchmark_realism_results/`), which is a system-ranking question rather than a "
        "discrimination one.\n")

    with open(os.path.join(C.OUT_DIR, "exp2_neg_discrimination.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(C.OUT_DIR, "exp2.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n[written] analysis/out/exp2_neg_discrimination.json, exp2.md")


if __name__ == "__main__":
    run()
