import json
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common
import evaluate
import exp_e1_bootstrap as e1
import exp_e1b_clustered as clustered
import exp_e2_realrec_ci as e2


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


@pytest.fixture
def arche(tmp_path):
    dims = common.ARCHE_DIMS
    meta, preds = [], []
    for i, value in enumerate([1, 2, 3, 4, 5, None]):
        meta.append({"id": str(i), "language": "en", "annotations": dict.fromkeys(dims, value)})
        preds.append({"id": str(i), "response": dict.fromkeys(dims, value or 1)})
    return write(tmp_path, "metadata.jsonl", meta), write(tmp_path, "preds.jsonl", preds)


def test_human_mask_and_imputation_are_separate():
    rec = {"annotations": {"raters": {"emotion_intensity": [None, "N/A", "", 3, "5"]}}}
    assert common.human_mean(rec, "emotion_intensity") == 4
    assert common.human_mean(rec, "emotion_intensity", "impute") == 2.2
    assert common.human_mean(rec, "absent") is None


@pytest.mark.parametrize("bad", ["not json", "[]", "null", '{"pitch_dynamics":"N/A"}',
                                 '{"pitch_dynamics":NaN}'])
def test_malformed_predictions_do_not_become_floor_scores(bad):
    assert common._parse_response(bad) == {}


def test_masked_score_and_sample_bootstrap(arche):
    meta, pred = arche
    result = evaluate.evaluate(meta, [pred], "archetype", n_boot=30)
    assert result["overall"]["mean_dimension_pearson"] == pytest.approx(1)
    assert result["overall"]["dimensions"]["audio_quality"]["n"] == 5
    assert result["overall"]["sample_bootstrap_95ci"] == pytest.approx([1, 1])
    assert result["language_macro_pearson"] == pytest.approx(1)


def test_partial_dimensions_cannot_claim_headline_average(tmp_path, arche):
    meta, _ = arche
    pred = write(tmp_path, "partial.jsonl", [{"id": str(i), "response": {"audio_quality": i+1}}
                                            for i in range(5)])
    result = evaluate.evaluate(meta, [pred], "archetype", n_boot=0)
    assert result["overall"]["mean_dimension_pearson"] is None
    assert result["overall"]["n_defined_dimensions"] == 1


def test_duplicate_prediction_cells_rejected(tmp_path):
    pred = write(tmp_path, "duplicate.jsonl", [{"id": "a", "response": {"audio_quality": 2}}] * 2)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate.load_predictions([pred], "archetype")


def test_long_form_expected_scores_are_not_integer_truncated(tmp_path):
    pred = write(tmp_path, "long.jsonl", [{"id": "a.wav", "specific_dimension": "pitch_dynamics",
                                         "probability_score": 3.75}])
    result, observed = evaluate.load_predictions([pred], "realism")
    assert result == {"a": {"pitch_variation": 3.75}}
    assert observed == {"a"}


def test_trials_averaged_and_missing_trials_not_imputed(tmp_path):
    a = write(tmp_path, "a.jsonl", [{"id": "a", "response": {"audio_quality": 2}}])
    b = write(tmp_path, "b.jsonl", [{"id": "a", "response": {"audio_quality": 4}}])
    c = write(tmp_path, "c.jsonl", [{"id": "a", "response": "malformed"}])
    result, _ = evaluate.load_predictions([a, b, c], "archetype")
    assert result["a"]["audio_quality"] == 3


def test_authentic_requires_explicit_disjoint_manifests(arche, tmp_path):
    meta, pred = arche
    with pytest.raises(ValueError, match="requires"):
        evaluate.evaluate(meta, [pred], "realism", split="authentic")
    manifest = write(tmp_path, "ids.jsonl", [{"id": "0"}])
    with pytest.raises(ValueError, match="overlap"):
        evaluate.evaluate(meta, [pred], "realism", "authentic", manifest, manifest)


def test_split_normalizes_wav_suffix(tmp_path):
    assert evaluate.load_ids(write(tmp_path, "ids.jsonl", [{"id": "a.wav"}])) == {"a"}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.load_ids(write(tmp_path, "dups.jsonl", [{"id": "a.wav"}, {"id": "a"}]))


def test_authentic_pool_filtered_before_scoring(tmp_path):
    meta, preds = [], []
    for i in range(6):
        rid = f"role-eval_v6_speaker_001_{i}"
        meta.append({"id": rid, "annotations": {"raters": dict.fromkeys(common.REALISM_DIMS, [i+1]*3)}})
        preds.append({"id": rid, "response": dict.fromkeys(common.PRED2REL, i+1)})
    metadata = write(tmp_path, "pool.jsonl", meta)
    predictions = write(tmp_path, "pool-pred.jsonl", preds)
    train = write(tmp_path, "train.jsonl", meta[:2])
    test = write(tmp_path, "test.jsonl", meta[2:])
    result = evaluate.evaluate(metadata, [predictions], "realism", "authentic", train, test, n_boot=0)
    assert result["overall"]["dimensions"]["pitch_variation"]["n"] == 4
    assert result["coverage"]["n_prediction_ids_outside_evaluation"] == 2
    assert result["within_speaker"]["mean_dimension_pearson"] == pytest.approx(1)


def test_cluster_bootstrap_keeps_repeated_draws():
    meta = {"a": {"original_id": "ShowA_01x01"}, "b": {"original_id": "ShowB_01x01"}}
    assert clustered.cluster_indices(["a", "b"], ["ShowA", "ShowA"], meta).tolist() == [0, 0]


def test_sample_bootstrap_shares_id_draw_across_dimensions():
    ids = np.array(["a", "b", "c", "d"])
    cells = {d: (ids, np.arange(4.), np.arange(4.), np.array(["spk"]*4)) for d in e2.DIMS}
    indices = e2.sample_indices(cells, np.random.default_rng(42))
    assert all(np.array_equal(indices[d], indices[e2.DIMS[0]]) for d in e2.DIMS)


def test_speaker_bootstrap_excludes_two_clip_speaker():
    ids = np.array(["a", "b", "c", "d", "e"])
    spk = np.array(["large"]*3 + ["tiny"]*2)
    cells = {d: (ids, np.arange(5.), np.arange(5.), spk) for d in e2.DIMS}
    filtered = e2.clusterable_cells(cells)
    assert all(list(row[0]) == ["a", "b", "c"] for row in filtered.values())


def test_paired_bootstrap_identical_models_has_zero_delta():
    ids = np.array(["a", "b", "c", "d"])
    x = np.array([1., 4., 2., 3.])
    cells = {"score": (ids, x, x)}
    point, low, high, _, n = e1.paired_delta_avg(cells, cells, ["score"], n=30)
    assert (point, low, high, n) == (0, 0, 0, 4)


def test_cli_emits_strict_json(arche, tmp_path):
    meta, pred = arche
    output = tmp_path / "scores.json"
    script = Path(evaluate.__file__)
    run = subprocess.run([sys.executable, str(script), "--metadata", str(meta), "--predictions",
                          str(pred), "--track", "archetype", "--n-boot", "0", "--output", str(output)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["gating"] == "mask"
    assert "NaN" not in output.read_text()


@pytest.mark.parametrize("filename", ["test_role.py", "test_arche.py"])
def test_missing_or_broken_audio_cannot_silently_become_ablation(tmp_path, filename):
    # Isolate audio I/O from model initialization so this regression needs no GPU.
    path = Path(__file__).resolve().parents[1] / "finetuning/src" / filename
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_audio")
    def broken_audio(_):
        raise OSError("broken audio")
    namespace = {"torch": SimpleNamespace(Tensor=object), "os": os,
                 "torchaudio": SimpleNamespace(load=broken_audio)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(FileNotFoundError):
        namespace["_get_audio"](str(tmp_path / "missing.wav"))
    present = tmp_path / "broken.wav"
    present.write_bytes(b"invalid audio")
    with pytest.raises(RuntimeError, match="Failed to load"):
        namespace["_get_audio"](str(present))


def test_training_ablation_filters_preserve_positive_items():
    path = Path(__file__).resolve().parents[1] / "finetuning/scripts/make_ablation_splits.py"
    namespace = {"__name__": "split_recipe"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    rows = [{"id": "a", "transcript": "hello"},
            {"id": "b", "processing_method": "weak_global_story", "transcript": "你好"},
            {"id": "c", "processing_method": "weak_local_story", "transcript": "hello"}]
    splits = namespace["make_splits"](rows)
    assert [r["id"] for r in splits["ablate_neg_posonly"]] == ["a"]
    assert [r["id"] for r in splits["ablate_neg_drop_wglobal"]] == ["a", "c"]
    assert [r["id"] for r in splits["xling_train_zh"]] == ["b"]
    with pytest.raises(ValueError, match="Unknown"):
        namespace["make_splits"]([{"processing_method": "unexpected"}])
