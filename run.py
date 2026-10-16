#!/usr/bin/env python3
"""Dataset -> prepared inputs -> inference/training -> masked scores.

Start with: python run.py options
An offline plumbing example is available with: python run.py demo --output runs/demo
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parent
DIMENSIONS = {
    "realism": ["pitch_dynamics", "rhythmic_naturalness", "stress_emphasis", "emotion_accuracy",
                "emotion_intensity", "emotional_dynamic_range", "voice_identity_matching",
                "trait_embodiment", "local_scene_fit", "global_story_fit"],
    "archetype": ["audio_quality", "human_likeness", "appropriateness"],
}
LABELS = dict(zip(DIMENSIONS["realism"], ["pitch_variation", "rhythmic_naturalness", "stress_emphasis",
    "emotion_accuracy", "emotion_intensity", "emotion_transition", "voice_identity_matching",
    "traits_embodiment", "local_scene_fit", "global_story_coherence"]))
RAW_LABELS = dict(zip(LABELS.values(), ["pitch_variation", "rhythmic_naturalness", "stress_and_emphasis",
    "emotion_accuracy", "emotion_intensity_control", "dynamic_range", "voice_identity_matching",
    "trait_embodiment", "local_story_fit", "global_story_coherence"]))
BACKENDS = {
    "qwen2audio": {"model": "Qwen/Qwen2-Audio-7B", "requirements": "local.txt"},
    "qwen25": {"model": "Qwen/Qwen2.5-Omni-7B", "requirements": "local.txt"},
    "gpt4o": {"model": None, "requirements": "cloud.txt"},
    "gemini25": {"model": None, "requirements": "cloud.txt"},
    "drame": {"model": None, "requirements": "local.txt"},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing {path}. A README-only dataset snapshot has no runnable payload; see docs/RUNNING.md.")
    text = path.read_text(encoding="utf-8")
    result = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not result or any(not isinstance(r, dict) for r in result):
        raise ValueError(f"Expected nonempty JSON/JSONL records: {path}")
    return result


def rid(row):
    value = row.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every dataset/manifest record requires a nonempty id")
    value = value.removesuffix(".wav")
    if Path(value).name != value or value in {".", ".."} or "\\" in value:
        raise ValueError("Dataset IDs must be filenames without directory components")
    return value


def index(records):
    result = {}
    for row in records:
        key = rid(row)
        if key in result:
            raise ValueError(f"Duplicate dataset/manifest ID: {key}")
        result[key] = row
    return result


def write_rows(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def make_input(row, track, audio):
    """Canonical release fields -> existing inference/training script fields."""
    ann = row.get("annotations", {})
    raters = ann.get("raters")
    if not isinstance(raters, dict):
        raise ValueError(f"{rid(row)} needs canonical annotations.raters; legacy demo JSON is a different schema")
    if track == "realism":
        required = ["profile", "char_style", "transcript"]
        if any(not isinstance(row.get(k), str) for k in required) or not isinstance(row.get("scene", {}).get("local"), str):
            raise ValueError(f"{rid(row)} needs profile, scene.local, char_style, and transcript")
        annotations = {raw: raters.get(canonical, []) for canonical, raw in RAW_LABELS.items()}
        annotations["semantic_match"] = raters.get("semantic_match", [])
        annotations["annotator_confidence_rating"] = raters.get("confidence", [])
        converted = {"dataset_name": "ROLE", "char_profile": row["profile"],
                     "local_scene": row["scene"]["local"], "char_style": row["char_style"],
                     "char_age": row.get("char_age", ""), "transcript": row["transcript"]}
    else:
        if not isinstance(row.get("prompt"), str):
            raise ValueError(f"{rid(row)} needs the canonical archetype prompt field")
        annotations = {d: raters.get(d, []) for d in DIMENSIONS[track]}
        # Release-native content_pass is a consensus boolean, not a fabricated rater list.
        if not isinstance(ann.get("content_pass"), bool):
            raise ValueError(f"{rid(row)} needs a boolean annotations.content_pass")
        annotations["content_pass"] = [ann["content_pass"]]
        converted = {"dataset_name": "Arche", "question": row["prompt"]}
    if any(not isinstance(v, list) for v in annotations.values()):
        raise ValueError(f"{rid(row)}: canonical rater annotations must be lists")
    converted.update({"id": rid(row), "original_id": row.get("original_id"),
                      "language": row.get("language"), "annotations": annotations,
                      "wav_path": str(audio) if audio else None})
    for key in ["processing_method", "processing_level", "processing_reason"]:
        if key in row:
            converted[key] = row[key]
    return converted


def resolve_audio(row, asset_root, overrides):
    if rid(row) in overrides:
        value = overrides[rid(row)].get("wav_path")
        if not value:
            raise ValueError("Audio-map records need id and wav_path")
        path = Path(value).expanduser()
    else:
        value = row.get("audio", {}).get("synthesized", {}).get("path")
        if not value:
            raise ValueError(f"{rid(row)} has no reconstructed waveform. Mel-only snapshots need reconstruction first: supply --audio-map or use a snapshot with reconstructions. Saved-prediction scoring can use --score-only.")
        path = Path(value)
    path = path if path.is_absolute() else asset_root / path
    if not path.is_file():
        raise ValueError(f"Missing reconstructed audio for {rid(row)}: {path}")
    expected = row.get("audio", {}).get("synthesized", {}).get("sha256")
    if rid(row) not in overrides and expected and digest(path) != expected:
        raise ValueError(f"Reconstructed audio checksum mismatch: {rid(row)}")
    return path.resolve()


def stage_wav(source, target):
    # APIs and all local loaders receive actual PCM WAV, even when the release is FLAC.
    try:
        with wave.open(str(source), "rb") as handle:
            if handle.getnframes() == 0:
                raise ValueError(f"Empty audio: {source}")
        shutil.copyfile(source, target)
        return
    except (wave.Error, EOFError):
        pass
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ValueError("FLAC/other audio conversion needs: pip install -r requirements/workflow.txt") from exc
    audio, rate = sf.read(source, always_2d=True)
    if len(audio) == 0:
        raise ValueError(f"Empty audio: {source}")
    sf.write(target, audio, rate, format="WAV", subtype="PCM_16")


def prepare(args):
    records = index([r for path in args.metadata for r in rows(path)])
    if any(r.get("perspective") != args.track for r in records.values()):
        raise ValueError("Metadata perspective must match --track; prepare tracks separately")
    train = index(rows(args.train_ids)) if args.train_ids else None
    test = index(rows(args.test_ids)) if args.test_ids else None
    if (train is None) != (test is None):
        raise ValueError("Provide both --train-ids and --test-ids")
    if train is not None and set(train) & set(test):
        raise ValueError("Training/test ID overlap")
    if args.protocol == "authentic" and (args.track != "realism" or train is None or args.purpose != "evaluate"):
        raise ValueError("Authentic evaluation requires realism, --train-ids, --test-ids, and --purpose evaluate")
    selected = set(index(rows(args.ids))) if args.ids else set(records)
    if test is not None:
        membership = set(train) if args.purpose == "train" else set(test)
        if args.ids and selected - membership:
            raise ValueError("Requested --ids are outside the selected training/test membership")
        selected = selected if args.ids else membership
    if not selected or selected - records.keys():
        raise ValueError("Selected IDs are empty or absent from supplied metadata; use exact subset manifests")
    records = {key: records[key] for key in sorted(selected)}
    if args.purpose == "train" and train is None and any(r.get("split") != "train" for r in records.values()):
        raise ValueError("Training requires records explicitly marked split=train; test data cannot be used")
    if args.purpose == "evaluate" and any(r.get("split") == "train" for r in records.values()):
        raise ValueError("Evaluation selection contains split=train records")
    overrides = index(rows(args.audio_map)) if args.audio_map else {}
    root = args.asset_root.resolve()
    # Validate all paths and schemas before creating a new run directory.
    resolved = {key: resolve_audio(row, root, overrides) if not args.score_only else None
                for key, row in records.items()}
    inputs = [make_input(row, args.track, resolved[key]) for key, row in records.items()]
    out = args.output.resolve()
    if out.exists():
        raise ValueError(f"Output already exists: {out}; choose a new run directory")
    out.mkdir(parents=True)
    try:
        if not args.score_only:
            (out / "audio").mkdir()
            for row in inputs:
                target = out / "audio" / (row["id"] + ".wav")
                stage_wav(Path(row["wav_path"]), target)
                row["wav_path"] = str(target)
        write_rows(out / "metadata.jsonl", records.values())
        write_rows(out / "inputs.jsonl", inputs)
        files = {name: digest(out / name) for name in ["metadata.jsonl", "inputs.jsonl"]}
        if not args.score_only:
            files.update({str(p.relative_to(out)): digest(p) for p in sorted((out / "audio").glob("*.wav"))})
        if train is not None:
            write_rows(out / "train_ids.jsonl", ({"id": key} for key in sorted(selected)) if args.purpose == "train" else train.values())
            write_rows(out / "test_ids.jsonl", test.values() if args.purpose == "train" else ({"id": key} for key in sorted(selected)))
            files.update({name: digest(out / name) for name in ["train_ids.jsonl", "test_ids.jsonl"]})
        manifest = {"schema_version": 1, "track": args.track, "purpose": args.purpose,
                    "protocol": args.protocol, "audio_ready": not args.score_only,
                    "n_records": len(inputs), "files": files,
                    "sources": [{"path": str(path.resolve()), "sha256": digest(path)} for path in args.metadata]}
        if args.audio_map:
            manifest["audio_map"] = {"path": str(args.audio_map.resolve()), "sha256": digest(args.audio_map)}
        (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception:
        shutil.rmtree(out)
        raise
    print(f"Prepared {len(inputs)} {args.track} records in {out}")
    return out


def bundle(path, purpose=None, audio=False):
    path = path.resolve()
    manifest = json.loads((path / "run.json").read_text())
    for name, expected in manifest["files"].items():
        if name.startswith("audio/") and not audio:
            continue
        if digest(path / name) != expected:
            raise ValueError(f"Prepared file changed: {name}; prepare a new run to retain provenance")
    if purpose and manifest["purpose"] != purpose:
        raise ValueError(f"This command requires a --purpose {purpose} preparation")
    if audio:
        if not manifest["audio_ready"]:
            raise ValueError("Score-only preparation has no model audio. Prepare with reconstructed waveforms first.")
        for row in rows(path / "inputs.jsonl"):
            if not Path(row["wav_path"]).is_file():
                raise ValueError(f"Missing prepared audio: {row['wav_path']}")
    return manifest


def infer_plan(args):
    work = args.workdir.resolve()
    manifest = bundle(work, "evaluate", audio=True)
    track = manifest["track"]
    if args.zero_audio or args.zero_text:
        if args.backend != "drame" or track != "realism":
            raise ValueError("Modality ablations require --backend drame and realism")
    output = work / "predictions" / args.name
    if output.exists():
        raise ValueError(f"Prediction run already exists: {output}; use a new --name")
    if args.examples:
        examples = bundle(args.examples, "train", audio=True)
        if examples["track"] != track or args.backend not in {"gpt4o", "gemini25"}:
            raise ValueError("Few-shot needs a same-track training bundle and gpt4o/gemini25")
        if set(index(rows(work / "inputs.jsonl"))) & set(index(rows(args.examples / "inputs.jsonl"))):
            raise ValueError("Few-shot examples overlap evaluation IDs")
    plans = []
    if args.backend == "drame":
        if args.trials != 1:
            raise ValueError("DRAME runs once per dimension; --trials applies to baseline runs")
        if not args.checkpoint or not args.checkpoint.is_dir():
            raise ValueError("DRAME inference requires --checkpoint pointing to a local full model or adapter directory")
        if not any((args.checkpoint / n).exists() for n in ["adapter_config.json", "config.json"]):
            raise ValueError("Checkpoint has neither adapter_config.json nor config.json")
        script = "test_role.py" if track == "realism" else "test_arche.py"
        for dim in DIMENSIONS[track]:
            filename = ("role" if track == "realism" else "arche") + f"_eval_{dim}_exact.json"
            command = [sys.executable, str(ROOT / "finetuning/src" / script), "--model_path", str(args.checkpoint.resolve()),
                       "--data_file", str(work / "inputs.jsonl"), "--out_file", str(output / filename),
                       "--evaluation_type", "specific_dimension", "--specific_dimension", dim,
                       "--batch_size", str(args.batch_size), "--use_gt_thinking", "false"]
            if args.base_model:
                command += ["--base_model_path", args.base_model]
            if track == "realism":
                command += ["--zero_audio", str(args.zero_audio).lower(), "--zero_text", str(args.zero_text).lower()]
            plans.append({"cwd": str(ROOT / "finetuning"), "argv": command})
    else:
        model = args.model or BACKENDS[args.backend]["model"]
        if not model:
            raise ValueError("Cloud inference needs --model with your available Gemini model or Azure deployment name")
        section = "fewshot" if args.examples else "zeroshot"
        prompt = ("archetype_" if track == "archetype" else "") + ("few_shot_combined" if args.examples else "combined")
        for trial in range(1, args.trials + 1):
            command = [sys.executable, str(ROOT / section / "pyscripts/eval.py"),
                       "--model_name", args.backend, "--model_tag", model,
                       "--data_path", str(work / "inputs.jsonl"), "--output_path", str(output / f"trial{trial}.jsonl"),
                       "--prompt_type", prompt, "--prompt_version", "v2", "--batch_size", str(args.batch_size)]
            if args.examples:
                command += ["--few_shot_examples_path", str(args.examples.resolve() / "inputs.jsonl"),
                            "--num_few_shot", str(args.num_examples)]
            plans.append({"cwd": str(ROOT / section), "argv": command})
    return output, plans


def require_packages(names):
    missing = []
    for name in names:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError("Missing packages: " + ", ".join(missing) + ". See requirements/ and docs/RUNNING.md.")


def execute_plan(output, plans, dry_run):
    for plan in plans:
        print(f"{plan['cwd']}: {shlex.join(plan['argv'])}")
    if dry_run:
        print("Dry run: no inference/training started")
        return
    output.mkdir(parents=True)
    (output / "commands.json").write_text(json.dumps(plans, indent=2) + "\n")
    for plan in plans:
        env = os.environ.copy()
        env["PYTHONPATH"] = plan["cwd"] + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(plan["argv"], cwd=plan["cwd"], env=env, check=True)


def infer(args):
    output, plans = infer_plan(args)
    if not args.dry_run:
        require_packages((["torch", "torchaudio", "transformers", "accelerate", "peft", "tqdm"]
                          if args.backend in {"drame", "qwen2audio", "qwen25"} else
                          ["tqdm", "pydantic", "openai" if args.backend == "gpt4o" else "google-genai"]))
        if args.backend == "gpt4o" and not all(os.getenv(k) for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_ENDPOINT"]):
            raise ValueError("Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_ENDPOINT for this Azure adapter")
        if args.backend == "gemini25" and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_CLOUD_PROJECT")):
            raise ValueError("Set GEMINI_API_KEY/GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT plus Vertex ADC")
    execute_plan(output, plans, args.dry_run)


def train(args):
    work = args.workdir.resolve()
    bundle(work, "train", audio=True)
    if not args.config.is_file():
        raise ValueError("Missing training YAML; start from finetuning/conf/sft_lora.yaml")
    command = [sys.executable, str(ROOT / "finetuning/src/train_sft.py"),
               "--yaml_config", str(args.config.resolve()), "--data_file", str(work / "inputs.jsonl"),
               "--out_dir", str(args.output.resolve()), "--enable_tensorboard", "false"]
    if args.base_model:
        command += ["--model_name_or_path", args.base_model]
    if args.output.exists():
        raise ValueError("Training output already exists; choose a new --output")
    if not args.dry_run:
        require_packages(["torch", "torchaudio", "transformers", "trl", "datasets", "accelerate", "peft", "pyyaml"])
    execute_plan(args.output.resolve(), [{"cwd": str(ROOT / "finetuning"), "argv": command}], args.dry_run)


def score(args):
    work = args.workdir.resolve()
    manifest = bundle(work, "evaluate")
    sys.path.insert(0, str(ROOT / "analysis"))
    import evaluate
    paths = args.predictions
    if args.name:
        directory = work / "predictions" / args.name
        paths = sorted(directory.glob("*.jsonl")) + sorted(directory.glob("*_exact.json"))
    if not paths:
        raise ValueError("Supply --name of a completed inference run or --predictions with saved result files")
    result = evaluate.evaluate(work / "metadata.jsonl", paths, manifest["track"], manifest["protocol"],
                              work / "train_ids.jsonl" if "train_ids.jsonl" in manifest["files"] else None,
                              work / "test_ids.jsonl" if "test_ids.jsonl" in manifest["files"] else None,
                              args.n_boot, args.seed)
    result["workflow"] = {"manifest_sha256": digest(work / "run.json"),
                          "n_prepared_records": manifest["n_records"],
                          "prediction_files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in paths],
                          "demo_fixture": manifest.get("demo_fixture", False)}
    output = args.output or work / (f"scores-{args.name}.json" if args.name else "scores.json")
    if output.exists():
        raise ValueError(f"Score output already exists: {output}; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluate.json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"Scores: {output}\nMean dimension Pearson: {result['overall']['mean_dimension_pearson']}")
    return output


def demo(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Demo output already exists; choose a new directory")
    source = output / "fixture"
    source.mkdir(parents=True)
    metadata, predictions = [], []
    for i in range(8):
        name = f"fixture_{i}"
        path = source / f"{name}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"".join(struct.pack("<h", int(1000*math.sin(2*math.pi*(220+i*20)*n/16000))) for n in range(1600)))
        value = (i % 5) + 1
        annotations = {"raters": {d: [value, value, value] for d in LABELS.values()}}
        annotations.update(dict.fromkeys(LABELS.values(), value))
        metadata.append({"id": name, "perspective": "realism", "split": "test", "language": "en" if i<4 else "zh",
                         "profile": "Fictional workflow fixture", "scene": {"local": "Synthetic smoke test"},
                         "char_style": "neutral", "transcript": "Fixture", "annotations": annotations,
                         "audio": {"synthesized": {"path": str(path)}}})
        predictions.append({"id": name, "response": dict.fromkeys(DIMENSIONS["realism"], (i*3 % 5)+1)})
    write_rows(source / "metadata.jsonl", metadata)
    write_rows(source / "fixture-predictions.jsonl", predictions)
    work = prepare(argparse.Namespace(metadata=[source / "metadata.jsonl"], asset_root=source,
                   track="realism", purpose="evaluate", protocol="benchmark", ids=None,
                   train_ids=None, test_ids=None, audio_map=None, score_only=False, output=output / "prepared"))
    p = work / "run.json"
    manifest = json.loads(p.read_text()); manifest["demo_fixture"] = True
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    result = score(argparse.Namespace(workdir=work, name=None, predictions=[source / "fixture-predictions.jsonl"],
                   output=None, n_boot=20, seed=12345))
    print(f"Offline fixture complete: {result}. Predictions were fixtures, not model judgments or benchmark results.")


def fetch(args):
    require_packages(["huggingface-hub"])
    from huggingface_hub import snapshot_download
    destination = Path(snapshot_download(repo_id=args.repo, repo_type="dataset", revision=args.revision,
                                         local_dir=str(args.output.resolve())))
    payloads = sorted(destination.rglob("metadata/*.jsonl"))
    if not payloads:
        raise ValueError(f"Downloaded documentation to {destination}, but found no metadata payload. A placeholder page is not a runnable dataset. See docs/RUNNING.md.")
    print("Available metadata files:\n" + "\n".join(str(p) for p in payloads))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def name(value):
    if not value or not all(c.isalnum() or c in "-_" for c in value):
        raise argparse.ArgumentTypeError("Use letters, numbers, hyphens, or underscores for run names")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("options", help="Show runnable paths and prerequisites")
    q = sub.add_parser("fetch", help="Download a dataset snapshot; fail clearly on README-only placeholders")
    q.add_argument("--repo", choices=["Anuttacon/speech-drame-evalbench", "ftshijt/speech-drame-evalbench"], required=True)
    q.add_argument("--revision", required=True, help="Dataset revision; prefer a commit hash")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("prepare", help="Validate canonical release data, map fields, and stage model-ready WAVs")
    q.add_argument("--metadata", nargs="+", type=Path, required=True)
    q.add_argument("--asset-root", type=Path, required=True, help="Root relative to which metadata audio paths resolve")
    q.add_argument("--track", choices=list(DIMENSIONS), required=True)
    q.add_argument("--purpose", choices=["train", "evaluate"], default="evaluate")
    q.add_argument("--protocol", choices=["benchmark", "authentic"], default="benchmark")
    q.add_argument("--ids", type=Path, help="Optional exact selected-ID JSONL")
    q.add_argument("--train-ids", type=Path)
    q.add_argument("--test-ids", type=Path)
    q.add_argument("--audio-map", type=Path, help="JSONL id/wav_path mapping for external reconstructions")
    q.add_argument("--score-only", action="store_true", help="No waveform needed; cannot be used for model inference/training")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("infer", help="Run a selected baseline or DRAME checkpoint on prepared data")
    q.add_argument("--workdir", type=Path, required=True)
    q.add_argument("--backend", choices=list(BACKENDS), required=True)
    q.add_argument("--name", type=name, required=True, help="New prediction run name")
    q.add_argument("--model", help="Baseline model ID or Azure deployment name")
    q.add_argument("--checkpoint", type=Path, help="Local DRAME adapter/full checkpoint")
    q.add_argument("--base-model", help="Optional local/HF base model override for an adapter")
    q.add_argument("--examples", type=Path, help="Separate prepared training bundle for few-shot examples")
    q.add_argument("--num-examples", type=positive, default=3)
    q.add_argument("--trials", type=positive, default=1)
    q.add_argument("--batch-size", type=positive, default=4)
    q.add_argument("--zero-audio", action="store_true")
    q.add_argument("--zero-text", action="store_true")
    q.add_argument("--dry-run", action="store_true", help="Validate inputs and show exact commands without starting a model")
    q = sub.add_parser("train", help="Fine-tune with a prepared training bundle and explicit YAML recipe")
    q.add_argument("--workdir", type=Path, required=True)
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--base-model")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("score", help="Score one saved prediction run against its prepared canonical metadata")
    q.add_argument("--workdir", type=Path, required=True)
    group = q.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", type=name)
    group.add_argument("--predictions", nargs="+", type=Path)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--seed", type=int, default=12345)
    q.add_argument("--output", type=Path)
    q = sub.add_parser("demo", help="Offline generated-fixture preparation/scoring; no model or benchmark claim")
    q.add_argument("--output", type=Path, required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command == "options":
            print("""Choose a path:
  demo                 Offline generated WAV + canonical metadata + saved fixture predictions -> scores
  fetch -> prepare     Dataset snapshot -> canonical metadata and reconstructed audio -> prepared inputs
  prepare --score-only -> score  Canonical labels + existing predictions; no audio/model needed
  prepare -> infer -> score     Baseline model or local DRAME checkpoint -> primary masked scores
  prepare --purpose train -> train  Labeled training data -> fine-tuned checkpoint

Inference backends: qwen2audio, qwen25 (local GPU); gpt4o (Azure), gemini25 (API/Vertex); drame (your checkpoint).
Use infer/train --dry-run to check inputs and inspect commands before loading a model.
Few-shot: infer --backend gpt4o|gemini25 --examples <separate prepared training directory>.
Mel-only data need reconstructed waveforms via --audio-map before inference. Saved predictions can be scored without audio.
README-only dataset pages cannot supply data; trained DRAME weights must be supplied until published.
Start with docs/RUNNING.md. Dependencies: requirements/workflow.txt, cloud.txt, local.txt, training.txt.""")
        else:
            globals()[args.command](args)
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
