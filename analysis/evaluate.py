"""Primary masked scoring for canonical metadata and saved judge/model predictions.

Run from the repository root: python analysis/evaluate.py --help.
No inference, API calls, or model dependencies are required.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import common
from exp_e1_bootstrap import arche_human, avg_ci, build_cells, realism_human
from realrec_v2_clean import within_speaker_avg


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"Expected JSON/JSONL records in {path}")
    return records


def record_id(record):
    value = record.get("id") or record.get("audio_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every record needs an id or audio_path")
    return Path(value).name.removesuffix(".wav")


def load_ids(path):
    records = read_records(path)
    ids = [record_id(r) for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID manifest is empty or contains duplicate IDs: {path}")
    return set(ids)


def validate_split(train, test):
    overlap = train & test
    if overlap:
        raise ValueError(f"Training/test overlap: {len(overlap)} IDs")


def load_predictions(paths, track):
    """Average available trials per ID/dimension; never replace parse failures with 1.

    Each input file is one trial or one dimension. Duplicate ID/dimension cells in
    the same file are errors. Separate files may contain repeated trial cells.
    """
    dims = common.ARCHE_DIMS if track == "archetype" else common.REALISM_DIMS
    acc = defaultdict(lambda: defaultdict(list))
    observed = set()
    for path in paths:
        seen = set()
        for record in read_records(path):
            rid = record_id(record)
            observed.add(rid)
            if "specific_dimension" in record:
                dim = common.PRED2REL.get(record["specific_dimension"], record["specific_dimension"])
                values = {dim: record.get("probability_score")}
            elif track == "archetype":
                values = common._parse_response_arche(record.get("response", record))
            else:
                raw = common._parse_response(record.get("response", record))
                values = {common.PRED2REL[key]: value for key, value in raw.items()}
            for dim, value in values.items():
                if dim not in dims:
                    continue
                key = (rid, dim)
                if key in seen:
                    raise ValueError(f"Duplicate prediction {key} in {path}")
                seen.add(key)
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                if np.isfinite(value):
                    acc[rid][dim].append(value)
    preds = {rid: {dim: float(np.mean(vals)) for dim, vals in row.items()}
             for rid, row in acc.items()}
    return preds, observed


def score(preds, meta, track, n_boot, seed):
    dims = common.ARCHE_DIMS if track == "archetype" else common.REALISM_DIMS
    human = arche_human if track == "archetype" else realism_human
    cells = build_cells(preds, meta, dims, human)
    per_dim = {}
    for dim in dims:
        ids, pred, truth = cells[dim]
        per_dim[dim] = {"pearson": common.pearson(pred, truth), "n": len(ids)}
    defined = [v["pearson"] for v in per_dim.values() if np.isfinite(v["pearson"])]
    complete = len(defined) == len(dims)
    result = {
        "dimensions": per_dim,
        "mean_dimension_pearson": float(np.mean(defined)) if complete else None,
        "n_defined_dimensions": len(defined),
        "n_expected_dimensions": len(dims),
    }
    # An incomplete set of dimensions must not masquerade as the paper's Avg.
    if n_boot and complete:
        _, lo, hi = avg_ci(cells, dims, n=n_boot, seed=seed)
        result["sample_bootstrap_95ci"] = [lo, hi]
    return result


def evaluate(metadata, predictions, track, split="benchmark", train_ids=None,
             test_ids=None, n_boot=2000, seed=12345):
    if n_boot < 0:
        raise ValueError("n_boot must be nonnegative")
    if split == "authentic" and (train_ids is None or test_ids is None):
        raise ValueError("Authentic evaluation requires --train-ids and --test-ids")
    if (train_ids is None) != (test_ids is None):
        raise ValueError("Supply both training and test manifests")
    if split == "authentic" and track != "realism":
        raise ValueError("The authentic-speech protocol uses the realism track")
    meta = {}
    for record in read_records(metadata):
        rid = record_id(record)
        if rid in meta:
            raise ValueError(f"Duplicate metadata ID: {rid}")
        meta[rid] = record
    if not meta:
        raise ValueError("Metadata is empty")
    audit = {}
    if train_ids is not None:
        train, test = load_ids(train_ids), load_ids(test_ids)
        validate_split(train, test)
        if test - meta.keys():
            raise ValueError("Test IDs are absent from the supplied metadata")
        meta = {rid: meta[rid] for rid in sorted(test)}
        audit = {"n_train_ids": len(train), "n_test_ids": len(test), "overlap": 0}
    preds, observed = load_predictions(predictions, track)
    used = {rid: row for rid, row in preds.items() if rid in meta}
    if not used:
        raise ValueError("No usable predictions match the evaluation metadata")
    result = {
        "track": track, "split": split, "gating": "mask",
        "statistic": "mean of dimension Pearson correlations",
        "n_boot": n_boot, "seed": seed, "split_audit": audit,
        "coverage": {
            "n_metadata_ids": len(meta), "n_observed_prediction_ids": len(observed),
            "n_usable_prediction_ids": len(used),
            "n_ids_without_usable_prediction": len(meta.keys() - used.keys()),
            "n_prediction_ids_outside_evaluation": len(observed - meta.keys()),
        },
        "overall": score(used, meta, track, n_boot, seed),
    }
    langs = sorted({r.get("language") for r in meta.values() if r.get("language")})
    result["per_language"] = {
        lang: score(used, {rid: r for rid, r in meta.items() if r.get("language") == lang},
                    track, n_boot, seed) for lang in langs
    }
    language_means = [r["mean_dimension_pearson"] for r in result["per_language"].values()]
    result["language_macro_pearson"] = (
        float(np.mean(language_means))
        if language_means and all(v is not None for v in language_means) else None
    )
    if split == "authentic":
        within, by_speaker = within_speaker_avg(used, meta, sorted(meta), gating="mask")
        result["within_speaker"] = {"mean_dimension_pearson": within,
                                    "by_speaker": by_speaker}
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Canonical release JSONL")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True,
                        help="Saved JSON/JSONL trial or dimension files")
    parser.add_argument("--track", choices=["realism", "archetype"], required=True)
    parser.add_argument("--split", choices=["benchmark", "authentic"], default="benchmark")
    parser.add_argument("--train-ids", type=Path, help="Training-input JSONL with id fields")
    parser.add_argument("--test-ids", type=Path, help="Clean test-input JSONL with id fields")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.metadata, args.predictions, args.track, args.split,
                          args.train_ids, args.test_ids, args.n_boot, args.seed)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
