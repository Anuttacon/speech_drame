"""
Shared utilities for the Tier-1 strengthening analyses of Speech-DRAME / DRAME-EvalBench.

All three Tier-1 experiments (human-agreement ceiling, positive/negative discrimination,
system-level ranking) are *offline* analyses over artifacts that already exist on disk:

  * Human labels (canonical)  : preprocess/{dataset_final,archetype_dataset_final}/to_release/metadata/*.jsonl
                                -> annotations.<dim> (mean) and annotations.raters.<dim> (per-rater list)
                                -> sample_type / negative_type / perturbed_components / processing_level / language
                                -> sfnd.system (archetype only)
  * Zero-shot judge preds     : role_play_baseline/zeroshot/results/<m>_<split>-test_zeroshot_*detailed*results*.jsonl
  * Few-shot judge preds       : role_play_baseline/fewshot/results/<m>_<split>-test_fewshot_detailed_results_trial*.jsonl
  * DRAME-Eval (Instruct-LoRA): role_play_eval/eval_model/results{,_arche}/<exp>/<{role,arche}_eval_<dim>_exact.json>
                                -> probability_score is the expectation-decoded continuous score used by the paper.

Primary results use human-label masking and the clean authentic-speech split.
The historical realism_real key is retained only for contamination diagnostics;
realism_real_v2 is the held-out evaluation. See README.md for artifact requirements.

This module intentionally has no side effects on import.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT = os.path.abspath(os.environ.get("SPEECH_DRAME_ROOT", Path(__file__).resolve().parents[1]))
REALISM_META = os.path.join(PROJECT, "preprocess/dataset_final/to_release/metadata")
ARCHE_META = os.path.join(PROJECT, "preprocess/archetype_dataset_final/to_release/metadata")
ZS_DIR = os.path.join(PROJECT, "role_play_baseline/zeroshot/results")
FS_DIR = os.path.join(PROJECT, "role_play_baseline/fewshot/results")
EVAL_DIR = os.path.join(PROJECT, "role_play_eval/eval_model")
OUT_DIR = os.path.abspath(os.environ.get("SPEECH_DRAME_OUT", Path(__file__).resolve().parent / "out"))

# --------------------------------------------------------------------------------------
# Dimensions and name mappings
# --------------------------------------------------------------------------------------
# The 10 realism sub-dimensions, in the order used by the paper's Tables 3-4.
REALISM_DIMS = [
    "pitch_variation", "rhythmic_naturalness", "stress_emphasis",
    "emotion_accuracy", "emotion_intensity", "emotion_transition",
    "voice_identity_matching", "traits_embodiment",
    "local_scene_fit", "global_story_coherence",
]
REALISM_LABELS = {
    "pitch_variation": "Pitch", "rhythmic_naturalness": "Rhythm", "stress_emphasis": "Stress",
    "emotion_accuracy": "Accuracy", "emotion_intensity": "Intensity", "emotion_transition": "Transition",
    "voice_identity_matching": "Identity", "traits_embodiment": "Traits",
    "local_scene_fit": "Local", "global_story_coherence": "Global",
}
# emotion gating: these two sub-dims are floored to 1 ('N/A') when Emotion Accuracy gate fails.
GATED_DIMS = {"emotion_intensity", "emotion_transition"}

ARCHE_DIMS = ["audio_quality", "human_likeness", "appropriateness"]
ARCHE_LABELS = {"audio_quality": "Audio Quality", "human_likeness": "Human Likeness",
                "appropriateness": "Appropriateness"}

# prediction-response key  ->  canonical to_release dimension
PRED2REL = {
    "pitch_dynamics": "pitch_variation",
    "rhythmic_naturalness": "rhythmic_naturalness",
    "stress_emphasis": "stress_emphasis",
    "emotion_accuracy": "emotion_accuracy",
    "emotion_intensity": "emotion_intensity",
    "emotional_dynamic_range": "emotion_transition",
    "voice_identity_matching": "voice_identity_matching",
    "trait_embodiment": "traits_embodiment",
    "local_scene_fit": "local_scene_fit",
    "global_story_fit": "global_story_coherence",
}
REL2PRED = {v: k for k, v in PRED2REL.items()}

# --------------------------------------------------------------------------------------
# Model registry. Each model maps to file patterns per split.
# split keys: 'realism_base' (in-distribution), 'realism_real' (real recordings), 'archetype'
# --------------------------------------------------------------------------------------
# zero-shot baseline file stems (prefix before _<split>-test_)
ZS_STEMS = {
    "Gemini2.5Pro": ("gemini-2.5-pro", True),    # (stem, proprietary -> 5 trials)
    "GPT4o-audio": ("gpt4ofull", True),
    "KimiAudio": ("kimiaudio", False),
    "Qwen2Audio": ("qwen2audio-7b", False),
    "Qwen2.5Omni": ("qwen25-7b", False),
}
# few-shot stems (only proprietary judges were run few-shot)
FS_STEMS = {
    "Gemini2.5Pro": "gemini-2.5-pro",
    "GPT4o-audio": "gpt-4o-audio-preview",
}
# baseline split token in filenames
ZS_SPLIT_TOKEN = {"realism_base": "all-test", "realism_real": "real-test", "archetype": "arche-test"}
FS_SPLIT_TOKEN = {"realism_base": "basic-test", "realism_real": "real-test", "archetype": "arche-test"}

# DRAME-Eval (final Instruct-LoRA) result dirs
DRAME_DIRS = {
    "realism_base": os.path.join(EVAL_DIR, "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"),
    "realism_real": os.path.join(EVAL_DIR, "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testreal"),
    # Clean real-rec eval: final Instruct-LoRA scored on the canonical held-out test_real_v2
    # (1,084 = qyf/zrr/zyx/yy/pro-mufei-2), produced by job-pqa5dxofkv. The 'realism_real'
    # dir above is the train-contaminated v1-list run (see handoffs/rebuttal_r1.md §3).
    "realism_real_v2": os.path.join(EVAL_DIR, "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_test_real_v2"),
    # Seen-speaker experiment: LoRA retrained on a per-speaker 80/20 utterance-disjoint
    # split (all 7 speakers incl pro-mufei in BOTH train and test), scored on the 432
    # held-out utterances. Tests within-speaker generalization for seen speakers.
    "perspk_seen": os.path.join(EVAL_DIR, "results/sft_lora_perspk_real_v2_checkpoint-1000_test_perspk"),
    # Matched speaker-DISJOINT control: same recipe/train-size as perspk_seen but whole-speaker
    # holdout (train 5 speakers, test yy+zyx). Isolates speaker-overlap as cause of the swing.
    "disjoint_ctrl": os.path.join(EVAL_DIR, "results/sft_lora_disjoint_real_v2_checkpoint-2000_test_disjoint"),
    "archetype": os.path.join(EVAL_DIR, "results_arche/sft_lora_addcontent_specific_dimension_checkpoint-10000"),
}

# canonical metadata file per split
META_FILE = {
    "realism_base": os.path.join(REALISM_META, "test_base.jsonl"),
    "realism_real": os.path.join(REALISM_META, "test_realrec.jsonl"),
    # test_realrec.jsonl (2,164) contains human ratings for all 1,084 test_real_v2 ids
    # (verified: 1084/1084 join), so the same metadata serves the clean split.
    "realism_real_v2": os.path.join(REALISM_META, "test_realrec.jsonl"),
    # test_realrec.jsonl (2,164) covers all real ids, incl. the 432 held-out per-speaker ids.
    "perspk_seen": os.path.join(REALISM_META, "test_realrec.jsonl"),
    "disjoint_ctrl": os.path.join(REALISM_META, "test_realrec.jsonl"),
    "archetype": os.path.join(ARCHE_META, "test.jsonl"),
}


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _norm_id(s):
    return s.replace(".wav", "") if isinstance(s, str) else s


def load_metadata(split):
    """Return {id -> record} from the canonical to_release metadata for a split."""
    out = {}
    with open(META_FILE[split]) as f:
        for line in f:
            r = json.loads(line)
            out[_norm_id(r["id"])] = r
    return out


def human_rater_arrays(rec, dim, gating="mask"):
    """Per-rater list for one canonical dimension from a metadata record.

    gating: 'mask' -> drop None / 'N/A' entries (clean ordinal protocol);
            'impute' -> map None/'N/A' to 1 (paper convention for gated emotion dims).
    Returns a list of ints (possibly shorter than n_raters under 'mask').
    """
    arr = rec.get("annotations", {}).get("raters", {}).get(dim)
    if not arr:
        return []
    if gating not in {"mask", "impute"}:
        raise ValueError("gating must be mask or impute")
    out = []
    for v in arr:
        if v is None or (isinstance(v, str) and v.strip().upper() in ("N/A", "")):
            if gating == "impute":
                out.append(1)
            # else: mask -> skip
            continue
        out.append(int(float(v)))
    return out


def human_mean(rec, dim, gating="mask"):
    arr = human_rater_arrays(rec, dim, gating=gating)
    return float(np.mean(arr)) if arr else None


def _parse_response(resp):
    """Parse a judge 'response' field (JSON string or dict) -> {pred_key: float}."""
    if not isinstance(resp, (str, dict)):
        return {}
    if isinstance(resp, dict):
        parsed = resp
    else:
        s = re.sub(r"```json\s*", "", resp)
        s = re.sub(r"```\s*", "", s).strip()
        try:
            parsed = json.loads(s)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for pk in PRED2REL:
        if pk in parsed:
            try:
                value = float(parsed[pk])
                if np.isfinite(value):
                    out[pk] = value
            except (TypeError, ValueError):
                pass
    return out


def load_judge_preds(model, split, shot="zero"):
    """Return {id -> {canonical_dim: score}} for a zero/few-shot judge on a split.

    Proprietary judges are averaged over their available trial files.
    Returns None if no prediction file exists (e.g. few-shot for open models).
    """
    if shot == "zero":
        stem, _ = ZS_STEMS[model]
        token = ZS_SPLIT_TOKEN[split]
        pat = os.path.join(ZS_DIR, f"{stem}_{token}_zeroshot_*detailed*results*.jsonl")
    else:
        if model not in FS_STEMS:
            return None
        stem = FS_STEMS[model]
        token = FS_SPLIT_TOKEN[split]
        pat = os.path.join(FS_DIR, f"{stem}_{token}_fewshot_detailed_results*.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for fp in files:
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rid = r.get("id")
                if not rid:
                    ap = r.get("audio_path", "")
                    rid = ap.split("/")[-1] if ap else ""
                rid = _norm_id(rid)
                for pk, v in _parse_response(r.get("response")).items():
                    acc[rid][PRED2REL[pk]].append(v)
    return {rid: {d: float(np.mean(vs)) for d, vs in dd.items()} for rid, dd in acc.items()}


def _parse_response_arche(resp):
    """Parse an archetype judge 'response' -> {audio_quality,human_likeness,appropriateness,content_pass}."""
    if not isinstance(resp, (str, dict)):
        return {}
    if isinstance(resp, dict):
        parsed = resp
    else:
        s = re.sub(r"```json\s*", "", resp)
        s = re.sub(r"```\s*", "", s).strip()
        try:
            parsed = json.loads(s)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for k in ARCHE_DIMS:
        if k in parsed:
            try:
                value = float(parsed[k])
                if np.isfinite(value):
                    out[k] = value
            except (TypeError, ValueError):
                pass
    if "content_pass" in parsed:
        cp = parsed["content_pass"]
        try:
            value = float(cp)
            if np.isfinite(value):
                out["content_pass"] = value
        except (TypeError, ValueError):
            pass
    return out


def load_judge_preds_arche(model, shot="zero"):
    """Return {id -> {arche_dim: score}} for a judge on the archetype test split."""
    if shot == "zero":
        stem, _ = ZS_STEMS[model]
        pat = os.path.join(ZS_DIR, f"{stem}_arche-test_zeroshot_*detailed*results*.jsonl")
    else:
        if model not in FS_STEMS:
            return None
        pat = os.path.join(FS_DIR, f"{FS_STEMS[model]}_arche-test_fewshot_detailed_results*.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for fp in files:
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rid = r.get("id")
                if not rid:
                    ap = r.get("audio_path", "")
                    rid = ap.split("/")[-1] if ap else ""
                rid = _norm_id(rid)
                for k, v in _parse_response_arche(r.get("response")).items():
                    acc[rid][k].append(v)
    return {rid: {k: float(np.mean(vs)) for k, vs in dd.items()} for rid, dd in acc.items()}


def load_drame_preds(split, dims=None):
    """Return {id -> {canonical_dim: probability_score}} for the final DRAME-Eval model."""
    d = DRAME_DIRS[split]
    if split == "archetype":
        files = {dim: os.path.join(d, f"arche_eval_{dim}_exact.json") for dim in (dims or ARCHE_DIMS)}
        keymap = {dim: dim for dim in (dims or ARCHE_DIMS)}
    else:
        wanted = dims or REALISM_DIMS
        files = {rd: os.path.join(d, f"role_eval_{REL2PRED[rd]}_exact.json") for rd in wanted}
        keymap = {rd: rd for rd in wanted}
    out = defaultdict(dict)
    for canon_dim, fp in files.items():
        if not os.path.exists(fp):
            continue
        for rec in json.load(open(fp)):
            out[_norm_id(rec["id"])][keymap[canon_dim]] = float(rec["probability_score"])
    return dict(out)


# --------------------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------------------
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return float(scipy.stats.spearmanr(x, y).correlation)


def kendall(x, y):
    if len(x) < 2:
        return float("nan")
    return float(scipy.stats.kendalltau(x, y).correlation)


def auroc(scores, labels):
    """AUROC of `scores` separating positive(label=1) from negative(label=0).

    Oriented so that higher score => more 'positive'. Uses the rank-sum identity.
    Returns nan if a class is empty.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    npos = int((labels == 1).sum())
    nneg = int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = scipy.stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def bootstrap_ci(values_fn, n=2000, seed=0, alpha=0.05):
    """Generic percentile bootstrap.

    values_fn(rng) -> a scalar statistic computed on a resample drawn using rng.
    Returns (lo, hi). Caller is responsible for resampling inside values_fn.
    """
    rng = np.random.default_rng(seed)
    stats = np.array([values_fn(rng) for _ in range(n)])
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def paired_pearson_ci(x, y, n=2000, seed=0):
    """Bootstrap CI for pearson(x, y) by resampling sample indices."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = len(x)

    def stat(rng):
        idx = rng.integers(0, m, m)
        return pearson(x[idx], y[idx])

    return bootstrap_ci(stat, n=n, seed=seed)


def krippendorff_alpha_ordinal(units):
    """Ordinal Krippendorff's alpha.

    units: list of per-item rater-value lists (variable length, missing already dropped).
    Implements the standard coincidence-matrix formulation with ordinal delta.
    Returns nan if undefined.
    """
    # collect set of possible values
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return float("nan")
    vidx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    # coincidence matrix
    coin = np.zeros((k, k), float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                coin[vidx[u[a]], vidx[u[b]]] += 1.0 / (m - 1)
    n_total = coin.sum()
    if n_total == 0:
        return float("nan")
    nc = coin.sum(axis=1)  # marginal per value

    # ordinal metric delta(c,k) = ( sum_{g=c..k} n_g - (n_c+n_k)/2 )^2
    def delta(ci, ki):
        lo, hi = (ci, ki) if ci <= ki else (ki, ci)
        s = nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2.0
        return s * s

    Do = 0.0
    De = 0.0
    for c in range(k):
        for kk in range(k):
            d = delta(c, kk)
            Do += coin[c, kk] * d
            De += nc[c] * nc[kk] * d
    Do /= n_total
    De /= (n_total * (n_total - 1))
    if De == 0:
        return float("nan")
    return float(1 - Do / De)


def fmt_ci(point, lo, hi, p=3):
    if point != point:  # nan
        return "n/a"
    return f"{point:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


# --------------------------------------------------------------------------------------
# Sanity reproduction (run: python analysis/common.py)
# --------------------------------------------------------------------------------------
def reproduce_sanity():
    """Check the primary masked final-model point estimates (full artifacts required)."""
    for split, reference in [("realism_base", 0.6020228921588984),
                             ("realism_real_v2", -0.14207966172733136),
                             ("archetype", 0.6288070586287393)]:
        meta = load_metadata(split)
        preds = load_drame_preds(split)
        if split == "realism_real_v2":
            from realrec_v2_clean import load_v2_ids
            keep = set(load_v2_ids())
            preds = {rid: row for rid, row in preds.items() if rid in keep}
        dims = ARCHE_DIMS if split == "archetype" else REALISM_DIMS
        rs = []
        for dim in dims:
            pairs = [(row[dim], human_mean(meta[rid], dim))
                     for rid, row in preds.items() if rid in meta and dim in row]
            pairs = [(p, g) for p, g in pairs if g is not None]
            rs.append(pearson(*zip(*pairs)) if pairs else float("nan"))
        value = float(np.mean(rs))
        if not np.isfinite(value) or not np.isclose(value, reference, atol=1e-6):
            raise ValueError(f"{split}: got {value}, expected {reference}; check artifacts")
        print(f"{split}: {value:.6f} (verified)")


if __name__ == "__main__":
    reproduce_sanity()
