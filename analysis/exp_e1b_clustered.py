"""Source-series clustered bootstrap with repeated draws preserved.

Uses per-dimension common IDs for paired comparisons. This corrects the interim
rebuttal implementation, which collapsed repeated cluster draws into a set.
Regenerated intervals can differ from the saved rebuttal intervals.
"""
from __future__ import annotations
from collections import defaultdict
import json
import os
import re
import numpy as np
import common
import exp_e1_bootstrap as e1

N_BOOT = 2000
SEED = 12345
DIMS = common.REALISM_DIMS
_EP = re.compile(r"_\d+x\d+")


def family_of(rid, meta):
    oid = meta.get(rid, {}).get("original_id") or rid
    match = _EP.search(oid)
    return oid[:match.start()] if match else common._norm_id(oid)


def aligned_pairs(cells_a, cells_b, dims):
    paired_a, paired_b = {}, {}
    for dim in dims:
        ia, pa, ga = cells_a[dim]
        ib, pb, gb = cells_b[dim]
        mb = {rid: (p, g) for rid, p, g in zip(ib, pb, gb)}
        shared = [(rid, p, g, *mb[rid]) for rid, p, g in zip(ia, pa, ga) if rid in mb]
        if any(not np.isclose(g, other_g) for _, _, g, _, other_g in shared):
            raise ValueError("Paired comparisons require identical human labels")
        ids = np.array([r[0] for r in shared])
        paired_a[dim] = (ids, np.array([r[1] for r in shared]), np.array([r[2] for r in shared]))
        paired_b[dim] = (ids, np.array([r[3] for r in shared]), np.array([r[4] for r in shared]))
    return paired_a, paired_b


def cluster_indices(ids, drawn, meta):
    """Repeat all rows each time their source cluster is drawn."""
    groups = defaultdict(list)
    for i, rid in enumerate(ids):
        groups[family_of(rid, meta)].append(i)
    return np.array([i for key in drawn for i in groups[key]], dtype=int)


def clustered_boot(cells_a, cells_b, meta, n=N_BOOT, seed=SEED):
    a, b = aligned_pairs(cells_a, cells_b, DIMS)
    ids = sorted(set().union(*(set(a[d][0]) for d in DIMS)))
    keys = sorted({family_of(rid, meta) for rid in ids})
    if not keys:
        raise ValueError("No common prediction IDs for clustered bootstrap")
    def avg(cells, drawn):
        rs = []
        for dim in DIMS:
            ii, pp, gg = cells[dim]
            idx = cluster_indices(ii, drawn, meta)
            rs.append(common.pearson(pp[idx], gg[idx]))
        return float(np.mean(rs))
    rng = np.random.default_rng(seed)
    av, bv = [], []
    for _ in range(n):
        drawn = rng.choice(keys, size=len(keys), replace=True)
        av.append(avg(a, drawn))
        bv.append(avg(b, drawn))
    av, bv = np.asarray(av), np.asarray(bv)
    delta = av - bv
    def ci(values):
        values = values[np.isfinite(values)]
        return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]
    valid = delta[np.isfinite(delta)]
    return {"n_clusters": len(keys), "n_samples": len(ids),
            "DRAME_avg_ci": ci(av), "baseline_avg_ci": ci(bv), "delta_ci": ci(delta),
            "delta_p_le0": float(np.mean(valid <= 0)) if len(valid) else None,
            "delta_point": avg(a, keys) - avg(b, keys)}


def main():
    meta, _, table = e1.build_realism_table("realism_base")
    results = {}
    for label in ["Gemini2.5Pro (FS)", "GPT4o-audio (FS)"]:
        if label in table:
            results[label] = clustered_boot(table["DRAME-Eval"][1], table[label][1], meta)
    os.makedirs(common.OUT_DIR, exist_ok=True)
    with open(os.path.join(common.OUT_DIR, "e1b_clustered.json"), "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
