"""Build deterministic E20 negative and E21 monolingual training manifests.

Ported from the saved rebuttal recipe. The language probe uses transcript script,
which is deliberately distinct from the main analysis's canonical language tags.
"""
import argparse
import json
from pathlib import Path


def is_zh(text):
    return any("一" <= char <= "鿿" for char in (text or ""))


def make_splits(rows):
    allowed = {None, "gt_setup", "weak_global_story", "weak_local_story"}
    if any(row.get("processing_method") not in allowed for row in rows):
        raise ValueError("Unknown processing_method; inspect the source before defining negatives")
    negative = {"weak_global_story", "weak_local_story"}
    return {
        "ablate_neg_full": rows,
        "ablate_neg_posonly": [r for r in rows if r.get("processing_method") not in negative],
        "ablate_neg_drop_wglobal": [r for r in rows if r.get("processing_method") != "weak_global_story"],
        "ablate_neg_drop_wlocal": [r for r in rows if r.get("processing_method") != "weak_local_story"],
        "xling_train_zh": [r for r in rows if is_zh(r.get("transcript"))],
        "xling_train_en": [r for r in rows if not is_zh(r.get("transcript"))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Model-input training JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the six splits")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    ids = [row["id"].removesuffix(".wav") for row in rows]
    if not rows or len(ids) != len(set(ids)):
        parser.error("Source is empty or contains duplicate IDs")
    try:
        splits = make_splits(rows)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for name, records in splits.items():
        directory = args.output_dir / name
        directory.mkdir()
        (directory / "data.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        print(f"{name}: {len(records)} records")


if __name__ == "__main__":
    main()
