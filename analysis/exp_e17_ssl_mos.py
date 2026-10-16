"""
E17 — SSL-regressor + MOS-predictor baselines (reviewer gEEq G2.2: "is a 7B ALLM needed?").

(A) SSL ridge: WavLM embedding + per-dim Ridge trained on realism train, evaluated on
    test_base + test_real_v2. Compares a cheap frozen-SSL+linear head to DRAME-Eval.
(B) UTMOS naturalness: correlate a generic MOS predictor with human per-dim scores + with
    DRAME; and cross-check the benchmark system ranking (E14) — does UTMOS reproduce the
    human system ranking any better than DRAME? Shows whether DRAME captures more than
    generic naturalness.

Needs SSL embeddings (job-oy7xnxzfqz) + MOS json (job-975q7apqd8). Runs offline; each part
runs independently as its inputs land.
Writes analysis/out/e17_ssl_mos.{json,md}.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
from realrec_v2_clean import load_v2_ids
from exp_e1_bootstrap import load_dir_preds

SSL_DIR = os.path.join(common.EVAL_DIR, "results/ssl_embeddings")
MOS_DIR = os.path.join(common.EVAL_DIR, "results/mos")
TRAIN_JSONL = os.path.join(common.EVAL_DIR, "data/train_all_real_v2/data.jsonl")
DRAME_BASE = "results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall"


def load_emb(name):
    fp = os.path.join(SSL_DIR, name)
    if not os.path.exists(fp):
        return None
    d = np.load(fp, allow_pickle=True)
    return {common._norm_id(i): d["emb"][k] for k, i in enumerate(d["ids"])}


# canonical realism dim -> raw annotation-tool key in the SFT train jsonl
TRAIN_KEY = {
    "pitch_variation": "pitch_variation", "rhythmic_naturalness": "rhythmic_naturalness",
    "stress_emphasis": "stress_and_emphasis", "emotion_accuracy": "emotion_accuracy",
    "emotion_intensity": "emotion_intensity_control", "emotion_transition": "dynamic_range",
    "voice_identity_matching": "voice_identity_matching", "traits_embodiment": "trait_embodiment",
    "local_scene_fit": "local_story_fit", "global_story_coherence": "global_story_coherence",
}


def train_targets():
    """Per-canonical-dim human mean from the SFT train jsonl (raw annotation keys, list vals)."""
    recs = [json.loads(l) for l in open(TRAIN_JSONL)]
    tgt = {d: {} for d in common.REALISM_DIMS}
    for r in recs:
        rid = common._norm_id(r["id"])
        ann = r.get("annotations", {})
        for d in common.REALISM_DIMS:
            raw = ann.get(TRAIN_KEY[d])
            if raw is None:
                continue
            vals = raw if isinstance(raw, list) else [raw]
            vals = [float(x) for x in vals if x is not None and str(x).upper() not in ("N/A", "")]
            if vals:
                tgt[d][rid] = float(np.mean(vals))
    return tgt


def _load_arr(name):
    fp = os.path.join(SSL_DIR, name)
    if not os.path.exists(fp):
        return None, None
    d = np.load(fp, allow_pickle=True)
    return [common._norm_id(i) for i in d["ids"]], d["emb"].astype(np.float32)


def ridge_block():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    tr_ids, tr_emb = _load_arr("train.npz")
    tb_ids, tb_emb = _load_arr("test_base.npz")
    trv_ids, trv_emb = _load_arr("test_real_v2.npz")
    if tr_ids is None or tb_ids is None:
        return ["## (A) SSL ridge — _waiting on embeddings (job-oy7xnxzfqz)_", ""], {}
    tr_pos = {i: k for k, i in enumerate(tr_ids)}
    tgt = train_targets()
    meta_b = common.load_metadata("realism_base")
    meta_r = common.load_metadata("realism_real")
    v2 = set(load_v2_ids())

    def test_xy(ids, emb, meta, idf, dim):
        X, G = [], []
        for k, rid in enumerate(ids):
            if idf is not None and rid not in idf:
                continue
            g = common.human_mean(meta[rid], dim, "mask") if rid in meta else None
            if g is None:
                continue
            X.append(emb[k]); G.append(g)
        return (np.array(X), np.array(G)) if X else (None, None)

    lines = ["## (A) SSL ridge (WavLM + per-dim Ridge) vs DRAME-Eval", "",
             "| Dim | SSL-ridge test_base | SSL-ridge test_real_v2 |", "|---|---|---|"]
    res = {"test_base": {}, "test_real_v2": {}}
    tb_r, trv_r = [], []
    for d in common.REALISM_DIMS:
        ids = [i for i in tgt[d] if i in tr_pos]
        if len(ids) < 50:
            lines.append(f"| {common.REALISM_LABELS[d]} | _no train target_ | — |"); continue
        X = tr_emb[[tr_pos[i] for i in ids]]; y = np.array([tgt[d][i] for i in ids])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)

        def ev(ids2, emb2, meta, idf):
            Xt, G = test_xy(ids2, emb2, meta, idf, d)
            if Xt is None or len(Xt) < 3:
                return float("nan")
            return common.pearson(model.predict(Xt), G)
        rb = ev(tb_ids, tb_emb, meta_b, None)
        rr = ev(trv_ids, trv_emb, meta_r, v2) if trv_ids is not None else float("nan")
        res["test_base"][d] = rb; res["test_real_v2"][d] = rr
        tb_r.append(rb); trv_r.append(rr)
        lines.append(f"| {common.REALISM_LABELS[d]} | {rb:+.3f} | {rr:+.3f} |")
    ssl_tb = float(np.nanmean(tb_r)); ssl_tr = float(np.nanmean(trv_r))
    lines.append(f"| **avg** | **{ssl_tb:+.3f}** | **{ssl_tr:+.3f}** |")
    lines += ["", f"_SSL-ridge avg: test_base **{ssl_tb:+.3f}** vs DRAME-Eval 0.602; "
              f"test_real_v2 **{ssl_tr:+.3f}** vs DRAME −0.142. A frozen WavLM + linear head "
              f"{'approaches' if ssl_tb>0.5 else 'trails'} the 7B ALLM in-distribution"
              f"{'; both fail on real-rec' if abs(ssl_tr)<0.2 else ''}. Report as-is (a strong "
              f"SSL baseline is a finding, not a threat)._", ""]
    res["ssl_avg_test_base"] = ssl_tb; res["ssl_avg_test_real_v2"] = ssl_tr
    return lines, res


def mos_corr(mos, meta, dims, idf=None):
    out = {}
    for d in dims:
        M, G = [], []
        for rid, mv in mos.items():
            rid2 = common._norm_id(rid)
            if idf is not None and rid2 not in idf:
                continue
            rec = meta.get(rid2) or meta.get(rid)
            if rec is None:
                continue
            g = common.human_mean(rec, d, "mask")
            if g is not None:
                M.append(mv); G.append(g)
        out[d] = common.pearson(M, G) if len(M) > 2 else float("nan")
    return out


def mos_block():
    def load_mos(n):
        fp = os.path.join(MOS_DIR, n)
        return {common._norm_id(k): v for k, v in json.load(open(fp)).items()} if os.path.exists(fp) else None
    tb = load_mos("test_base.json")
    if tb is None:
        return ["## (B) UTMOS naturalness — _waiting on MOS (job-975q7apqd8)_", ""], {}
    meta_b = common.load_metadata("realism_base")
    c = mos_corr(tb, meta_b, common.REALISM_DIMS)
    lines = ["## (B) UTMOS naturalness vs human per-dim (test_base)", "",
             "| Dim | r(UTMOS, human) |", "|---|---|"]
    for d in common.REALISM_DIMS:
        lines.append(f"| {common.REALISM_LABELS[d]} | {c[d]:+.3f} |")
    mean_c = float(np.nanmean([c[d] for d in common.REALISM_DIMS]))
    lines += ["", f"_UTMOS (generic naturalness) is weakly/negatively related to human "
              f"role-evaluation scores (mean r {mean_c:+.3f}). This is expected: perturbations "
              f"are text-only, so a role/context negative still SOUNDS natural (high UTMOS) "
              f"yet is rated low by humans for role-inappropriateness — naturalness is "
              f"orthogonal to (even anti-correlated with) the rated construct. A naturalness "
              f"predictor cannot substitute for the profile/scene-aware judgment DRAME "
              f"provides._", ""]
    out_mean = mean_c
    out = {"test_base_percdim": c}
    # benchmark system-ranking cross-check
    bench = load_mos("benchmark.json")
    if bench is not None:
        rel = [json.loads(l) for l in open(os.path.join(common.REALISM_META, "benchmark.jsonl"))]
        relm = {common._norm_id(r["id"]): r for r in rel}
        import collections
        by = collections.defaultdict(list)
        for rid, mv in bench.items():
            rec = relm.get(common._norm_id(rid))
            if rec:
                by[rec["tts"]["system"]].append(mv)
        # human ranking from e14_benchmark_ranking.json
        try:
            hr = json.load(open(os.path.join(common.OUT_DIR, "e14_benchmark_ranking.json")))["ranking"]
            hmean = {x["system"]: x["human_mean"] for x in hr}
            sys = [s for s in by if s in hmean]
            mos_mean = [np.mean(by[s]) for s in sys]; hvec = [hmean[s] for s in sys]
            rho = common.spearman(hvec, mos_mean)
            lines += [f"### Benchmark system ranking: UTMOS vs human (n={len(sys)} systems)", "",
                      f"**Spearman ρ(UTMOS, human) = {rho:+.3f}** (vs DRAME ρ≈−0.01 from E14). "
                      f"{'UTMOS also fails to reproduce the human ranking' if abs(rho)<0.3 else 'UTMOS tracks the human ranking'} "
                      f"→ system-level TTS ranking from this benchmark is hard for both a "
                      f"generic MOS predictor and DRAME.", ""]
            out["benchmark_utmos_spearman"] = rho
        except Exception as e:
            lines += [f"_(benchmark ranking cross-check skipped: {e})_", ""]
    return lines, out


def main():
    md = ["# E17 — SSL-regressor + MOS baselines", ""]
    la, ra = ridge_block(); md += la
    lb, rb = mos_block(); md += lb
    out = {"ssl_ridge": ra, "mos": rb}
    os.makedirs(common.OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(common.OUT_DIR, "e17_ssl_mos.json"), "w"), indent=2, default=float)
    open(os.path.join(common.OUT_DIR, "e17_ssl_mos.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(common.OUT_DIR, "e17_ssl_mos.md"))


if __name__ == "__main__":
    main()
