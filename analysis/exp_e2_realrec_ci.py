"""E2-realrec — Bootstrap CIs for the clean held-out real-recording result.

Fills the two items flagged as pending in the rebuttal:
  (1) DRAME-Eval real-recording PER-DIMENSION and Avg-10 bootstrap CIs on the clean
      held-out test_real_v2 (1,084, disjoint speakers). Uses the predictions already on
      disk (job-pqa5dxofkv); no re-inference.
  (2) The PAIRED real-rec difference CI, DRAME-Eval - best baseline (GPT4o-audio few-shot),
      on the common id set, so we can state whether the two are separable on real speech.

Two resampling schemes are reported:
  - sample-level: resample the N clip ids with replacement.
  - speaker-clustered: resample the 4 held-out speakers with replacement (the relevant
    cluster axis for real recordings, per the between-speaker decomposition).

Real-rec gated-dim handling is moot (impute == mask on this split; see summary), we use
impute to match the paper convention. Run from analysis/:  python3 exp_e2_realrec_ci.py
"""
from __future__ import annotations
import json, os, collections
import numpy as np
import common
from realrec_v2_clean import load_v2_ids as validated_v2_ids

N_BOOT = 2000
SEED = 12345
DIMS = common.REALISM_DIMS
GATING = "impute"  # == mask on real-rec (no None entries survive differently); paper convention
V2_DATA = os.path.join(common.EVAL_DIR, "data/test_real_v2/data.jsonl")
OUT = common.OUT_DIR


def speaker_of(rid: str) -> str:
    p = rid.replace("role-eval_", "").split("_")
    return p[1] if p and p[0].startswith("v") and p[0][1:].isdigit() else p[0]


def load_v2_ids():
    return set(validated_v2_ids())


def build_cells(preds, meta, keep_ids):
    """dim -> (ids[np.str], P[np.f], G[np.f], spk[np.str]) over ids in keep_ids w/ human+pred."""
    cells = {}
    for d in DIMS:
        I, P, G, S = [], [], [], []
        for rid in sorted(keep_ids):
            pr = preds.get(rid)
            if not pr or d not in pr or rid not in meta:
                continue
            g = common.human_mean(meta[rid], d, gating=GATING)
            if g is None:
                continue
            I.append(rid); P.append(pr[d]); G.append(g); S.append(speaker_of(rid))
        cells[d] = (np.array(I), np.array(P, float), np.array(G, float), np.array(S))
    return cells


def avg10(cells, idx_by_dim):
    rs = []
    for d in DIMS:
        _, P, G, _ = cells[d]
        idx = idx_by_dim[d]
        if len(idx) < 3:
            continue
        r = common.pearson(P[idx], G[idx])
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def ci(vals, lo=2.5, hi=97.5):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if len(v) else (float("nan"),)*2


def sample_indices(cells, rng):
    """Draw IDs once, then preserve their multiplicity in every dimension."""
    union = sorted(set().union(*(set(cells[d][0]) for d in DIMS)))
    drawn = rng.choice(union, size=len(union), replace=True)
    result = {}
    for d in DIMS:
        lookup = {rid: i for i, rid in enumerate(cells[d][0])}
        result[d] = np.array([lookup[rid] for rid in drawn if rid in lookup], dtype=int)
    return result


def clusterable_cells(cells):
    """Match the four-speaker protocol; exclude the two-clip speaker from clustering."""
    members = collections.defaultdict(set)
    for ids, _, _, speakers in cells.values():
        for rid, speaker in zip(ids, speakers):
            members[speaker].add(rid)
    keep = {speaker for speaker, ids in members.items() if len(ids) >= 3}
    return {d: tuple(v[np.isin(row[3], list(keep))] for v in row)
            for d, row in cells.items()}


def boot_perdim(cells, rng, clustered):
    """Return {dim -> (point, lo, hi)} and Avg CI, for one model. Shared resample across dims
    uses the same speaker/ID draws in every dimension."""
    if clustered:
        cells = clusterable_cells(cells)
    # point
    point = {d: common.pearson(cells[d][1], cells[d][2]) for d in DIMS}
    avg_pt = float(np.nanmean([point[d] for d in DIMS]))
    # bootstrap
    perdim = {d: [] for d in DIMS}
    avgs = []
    # precompute speaker->indices per dim for clustered
    for _ in range(N_BOOT):
        idx_by_dim = {}
        if clustered:
            # draw speakers once; apply same speaker set to every dim
            spk_universe = np.unique(np.concatenate([cells[d][3] for d in DIMS]))
            drawn = rng.choice(spk_universe, size=len(spk_universe), replace=True)
            for d in DIMS:
                S = cells[d][3]
                parts = [np.where(S == s)[0] for s in drawn]
                idx_by_dim[d] = np.concatenate(parts) if parts else np.array([], int)
        else:
            # Sample-level: one ID draw shared by every dimension.
            idx_by_dim = sample_indices(cells, rng)
        for d in DIMS:
            idx = idx_by_dim[d]
            r = common.pearson(cells[d][1][idx], cells[d][2][idx]) if len(idx) >= 3 else float("nan")
            perdim[d].append(r)
        avgs.append(avg10(cells, idx_by_dim))
    out = {d: (point[d],) + ci(perdim[d]) for d in DIMS}
    return out, (avg_pt,) + ci(avgs)


def boot_paired(cellsA, cellsB, rng, clustered):
    """Paired Δ = Avg10(A) - Avg10(B) on the COMMON id set per dim. Speaker-clustered or sample."""
    if clustered:
        cellsA, cellsB = clusterable_cells(cellsA), clusterable_cells(cellsB)
    # restrict each dim to common ids
    common_cells_A, common_cells_B = {}, {}
    for d in DIMS:
        iA, pA, gA, sA = cellsA[d]
        iB, pB, gB, sB = cellsB[d]
        setB = {i: k for k, i in enumerate(iB)}
        keep = [(k, setB[i]) for k, i in enumerate(iA) if i in setB]
        if not keep:
            common_cells_A[d] = (np.array([]), np.array([]), np.array([]), np.array([]))
            common_cells_B[d] = (np.array([]), np.array([]), np.array([]), np.array([]))
            continue
        ka = np.array([k for k, _ in keep]); kb = np.array([k for _, k in keep])
        common_cells_A[d] = (iA[ka], pA[ka], gA[ka], sA[ka])
        common_cells_B[d] = (iB[kb], pB[kb], gB[kb], sB[kb])
    a_pt = float(np.nanmean([common.pearson(common_cells_A[d][1], common_cells_A[d][2]) for d in DIMS]))
    b_pt = float(np.nanmean([common.pearson(common_cells_B[d][1], common_cells_B[d][2]) for d in DIMS]))
    deltas, a_s, b_s = [], [], []
    spk_universe = np.unique(np.concatenate([common_cells_A[d][3] for d in DIMS if len(common_cells_A[d][3])]))
    for _ in range(N_BOOT):
        idxA, idxB = {}, {}
        if clustered:
            drawn = rng.choice(spk_universe, size=len(spk_universe), replace=True)
            for d in DIMS:
                SA = common_cells_A[d][3]
                sel = np.concatenate([np.where(SA == s)[0] for s in drawn]) if len(SA) else np.array([], int)
                idxA[d] = sel; idxB[d] = sel  # paired: same rows for A and B
        else:
            idxA = sample_indices(common_cells_A, rng)
            idxB = idxA
        va = avg10(common_cells_A, idxA); vb = avg10(common_cells_B, idxB)
        a_s.append(va); b_s.append(vb); deltas.append(va - vb)
    n_common = sum(len(common_cells_A[d][0]) for d in DIMS) // len(DIMS)
    return {
        "drame_avg": (a_pt,) + ci(a_s),
        "baseline_avg": (b_pt,) + ci(b_s),
        "delta": (a_pt - b_pt,) + ci(deltas),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0)))),
        "n_common_per_dim": int(n_common),
    }


def main():
    meta = common.load_metadata("realism_real_v2")
    v2_ids = load_v2_ids()
    drame = common.load_drame_preds("realism_real_v2")
    base = common.load_judge_preds("GPT4o-audio", "realism_real", shot="few")  # best real-rec baseline

    drame_cells = build_cells(drame, meta, v2_ids & set(drame.keys()))
    # baseline: only keep ids that are BOTH in v2 clean set and have baseline preds
    base_cells = build_cells(base, meta, v2_ids & set(base.keys())) if base else None

    rng = np.random.default_rng(SEED)
    d_sample, d_sample_avg = boot_perdim(drame_cells, rng, clustered=False)
    rng = np.random.default_rng(SEED + 1)
    d_clust, d_clust_avg = boot_perdim(drame_cells, rng, clustered=True)

    result = {
        "n_ids_drame": int(len(drame_cells[DIMS[0]][0])),
        "gating": GATING,
        "drame_perdim_sample_level": {common.REALISM_LABELS[d]: d_sample[d] for d in DIMS},
        "drame_perdim_speaker_clustered": {common.REALISM_LABELS[d]: d_clust[d] for d in DIMS},
        "drame_avg_sample_level": d_sample_avg,
        "drame_avg_speaker_clustered": d_clust_avg,
    }
    if base_cells is not None:
        rng = np.random.default_rng(SEED + 2)
        paired_sample = boot_paired(drame_cells, base_cells, rng, clustered=False)
        rng = np.random.default_rng(SEED + 3)
        paired_clust = boot_paired(drame_cells, base_cells, rng, clustered=True)
        result["n_ids_baseline"] = int(len(base_cells[DIMS[0]][0]))
        result["paired_diff_sample_level"] = paired_sample
        result["paired_diff_speaker_clustered"] = paired_clust

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e2_realrec_ci.json"), "w") as f:
        json.dump(result, f, indent=2)

    # human-readable
    def fmt(t): return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    lines = ["# E2 — clean real-recording (test_real_v2) bootstrap CIs\n",
             f"DRAME-Eval ids scored: {result['n_ids_drame']}  (gating={GATING}; impute==mask on real-rec)\n",
             "## DRAME-Eval per-dimension Pearson (point [95% CI])\n",
             "| Dim | sample-level | speaker-clustered |", "|---|---|---|"]
    for d in DIMS:
        lines.append(f"| {common.REALISM_LABELS[d]} | {fmt(d_sample[d])} | {fmt(d_clust[d])} |")
    lines.append(f"| **Avg-10** | **{fmt(d_sample_avg)}** | **{fmt(d_clust_avg)}** |\n")
    if base_cells is not None:
        ps, pc = result["paired_diff_sample_level"], result["paired_diff_speaker_clustered"]
        lines += ["## Paired real-rec difference (DRAME-Eval − GPT4o-audio few-shot), common ids\n",
                  f"Common ids: {ps['n_common_per_dim']}\n",
                  "| Quantity | sample-level | speaker-clustered |", "|---|---|---|",
                  f"| DRAME Avg | {fmt(ps['drame_avg'])} | {fmt(pc['drame_avg'])} |",
                  f"| GPT4o-FS Avg | {fmt(ps['baseline_avg'])} | {fmt(pc['baseline_avg'])} |",
                  f"| **Δ (DRAME − base)** | **{fmt(ps['delta'])}** | **{fmt(pc['delta'])}** |",
                  f"| p (two-sided) | {ps['p_two_sided']:.3f} | {pc['p_two_sided']:.3f} |"]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "e2_realrec_ci.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
