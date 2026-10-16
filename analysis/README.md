# Rebuttal analyses and primary scoring

For dataset acquisition, metadata-to-input conversion, model execution, and training,
start with [the runnable workflow](../docs/RUNNING.md) and `python run.py options`.
This page documents scoring and the advanced saved-experiment analyses.


`evaluate.py` is the primary scorer for the revised paper. It reads saved
predictions and canonical release metadata, masks missing human labels, preserves
continuous expectation-decoded scores, and reports counts alongside each
dimension's Pearson correlation. The headline statistic is the arithmetic mean of
all 10 realism or all three archetype dimension correlations. An undefined
dimension makes the headline average `null`, rather than silently averaging a
smaller set. Content Pass accuracy is separate from the archetype correlation
average; the existing archetype parsers retain that historical analysis.

## Install and run

Python 3.10 or later is required. Scoring needs only NumPy and SciPy:

```bash
pip install -r analysis/requirements.txt
python analysis/evaluate.py \
  --track realism \
  --metadata /path/to/metadata/test_base.jsonl \
  --predictions /path/to/results/role_eval_*_exact.json \
  --output analysis/out/realism.json
```

For archetype, use `--track archetype`, the canonical archetype `test.jsonl`, and
`arche_eval_*_exact.json`. Saved baseline JSONL trial files can be supplied with
the same `--predictions` option. The scorer averages available valid trial scores
per ID and dimension, excludes malformed model responses, and reports the number
of IDs without usable predictions. A missing model response never becomes score 1.
Each file is one trial or dimension; duplicate ID/dimension cells within a file
are rejected. Supply only the intended trial files, without duplicated copies.

Canonical realism annotations use `annotations.raters.<canonical_dimension>`;
archetype uses the canonical masked scalar at `annotations.<dimension>`. The
small legacy demonstration files in `data/` use a different annotation schema;
they are inference examples, not replacements for canonical analysis metadata.

Authentic speech requires both model-input manifests, each containing `id` fields:

```bash
python analysis/evaluate.py \
  --track realism --split authentic \
  --metadata /path/to/metadata/test_realrec.jsonl \
  --train-ids /path/to/data/train_all_real_v2/data.jsonl \
  --test-ids /path/to/data/test_real_v2/data.jsonl \
  --predictions /path/to/clean-results/role_eval_*_exact.json \
  --output analysis/out/authentic.json
```

The scorer checks duplicate IDs, rejects training/test overlap after normalizing
`.wav` suffixes, requires every test ID to exist in metadata, and filters the pool
to the supplied test IDs before scoring. The 2,164-record pool includes 1,080
adaptation and 1,084 held-out records. It is not a 2,164-record test set. The
within-speaker result averages the dimension correlations within each eligible
speaker, then weights speakers equally; speakers with fewer than three clips are
excluded from that statistic. The pooled result retains all held-out clips.

The default sample bootstrap uses 2,000 resamples and seed 12345, drawing shared
IDs across dimensions. Set `--n-boot 0` for point estimates only. Per-language
results use canonical metadata tags. Strict JSON uses `null` for undefined
correlations. It reports coverage rather than asserting that a partial public
subset reproduces the full-corpus paper results.

## Imported experiment code

The other scripts were recovered from the saved rebuttal workspace. Original
source SHA-256 hashes are recorded in `imported-source-sha256.json`; that file
identifies the originals before the portability and correctness changes in this
PR. Internal experiment labels and some historical comments retain the rebuttal
draft's table numbering. Cluster submission scripts, saved prediction payloads,
private metadata, media, checkpoints, and credentials are not bundled here.

Set the artifact root before running these scripts:

```bash
export SPEECH_DRAME_ROOT=/path/to/experimental-artifacts
export SPEECH_DRAME_OUT=/path/to/new-analysis-output
python analysis/common.py
python analysis/exp_e5_gating.py
python analysis/exp_e3_perlang.py
```

The artifact root preserves the experiment directory layout:

```text
preprocess/dataset_final/to_release/metadata/
preprocess/archetype_dataset_final/to_release/metadata/
role_play_baseline/zeroshot/results/
role_play_baseline/fewshot/results/
role_play_eval/eval_model/data/
role_play_eval/eval_model/results/
role_play_eval/eval_model/results_arche/
```

`SPEECH_DRAME_ROOT` defaults to this repository, and `SPEECH_DRAME_OUT` defaults
to `analysis/out/` beside these scripts. No developer-specific absolute path is
required. Missing artifacts cause an error or an explicit unavailable/partial
result in the original experiment drivers. Do not treat a partial driver output
as a reproduced table. `common.py` checks the three primary point estimates and
raises if they do not match the verified full-corpus values.

| Analysis | Script(s) |
| --- | --- |
| Primary masked CIs and paired tests | `exp_e1_bootstrap.py`, `exp_e1c_archetype_paired.py` |
| Source-series clustered uncertainty | `exp_e1b_clustered.py` |
| Clean authentic speech and speaker decomposition | `realrec_v2_clean.py`, `exp_e2_realrec_ci.py`, `exp_e2b_human_within_speaker.py` |
| Per-language/domain/system breakdowns | `exp_e3_perlang.py`, `exp_e4_domain.py`, `exp_e15_archetype_persystem.py` |
| Mask versus human-only imputation sensitivity | `exp_e5_gating.py` |
| Human reliability and ordinal weighted kappa | `human_ceiling.py`, `exp_e6_weighted_kappa.py` |
| Ranking, calibration, negatives, leakage, and errors | `system_ranking.py`, `neg_discrimination.py`, `exp_e7_rank_calib.py` through `exp_e10_error_analysis.py` |
| Silence, gold-transcript text-only, and audio-only controls | `exp_e11_ablation.py`, `exp_e11_strat.py`, `exp_e11_textonly.py`, `exp_e19_textablation.py` |
| Provenance and SSL real/TTS probe | `exp_e12a_real_tts.py`, `exp_e12b_probe.py` |
| Metadata/source and unseen-system controls | `exp_e13_metadata_baselines.py`, `exp_e13b_seen_unseen.py`, `exp_e15_loso.py` |
| Generation-system ranking and semantic-match audit | `exp_e14_benchmark_ranking.py`, `exp_e14_model.py`, `exp_e16_semantic_match.py` |
| SSL/MOS and vocoder fidelity | `exp_e17_ssl_mos.py`, `exp_e18_vocoder.py` |
| Training negatives and monolingual transfer | `exp_ablation_xling.py` |
| Demographic, labeling, training-history, and voice-gender audits | `exp_r3a_demographic.py` through `exp_r3d_voice_gender.py` |
| Seen-speaker/disjoint retraining diagnostics | `realrec_perspk_eval.py`, `realrec_disjoint_contrast.py` |

E12b and E17 additionally need scikit-learn (the original experiment used 1.6.1)
and the saved SSL embeddings/MOS outputs. Provenance and demographic analyses
need their original source manifests and metadata fields. The negative and
cross-language analysis reads six saved 2,000-step retraining outputs. These
analysis scripts do not themselves reproduce the upstream feature extraction,
API inference, or GPU training jobs.

## Checkpoints and training controls

The final realism result directory is
`results/sft_lora_all-real-v2_specific_dimension_checkpoint-10000_testall`;
its clean authentic counterpart ends in `_test_real_v2`. The final archetype
directory is
`results_arche/sft_lora_addcontent_specific_dimension_checkpoint-10000`.
The similarly named archetype directory without `addcontent` is a different run.
The `common.py` key `realism_real` identifies the old contaminated model output
for diagnostics only; use `realism_real_v2` for the primary authentic result.

`finetuning/src/test_role.py` now includes the exact rebuttal `--zero_audio true`
(0.5 seconds of silence) and `--zero_text true` (blank profile, scene, and traits)
controls, and records both flags in saved inference output. Missing or unreadable
audio now raises an error in both fine-tuning inference entrypoints. These flags isolate modality contributions and are distinct from the
separately trained gold-transcript text-only judge. Both default to false.
Training uses non-reentrant gradient checkpointing by default, with a YAML
override, matching the DDP correction used for the rebuttal runs.

The deterministic negative/monolingual training filters are available as:

```bash
python finetuning/scripts/make_ablation_splits.py \
  --source /path/to/data/train_all_real_v2/data.jsonl \
  --output-dir /path/to/new-ablation-data
```

They preserve the original `processing_method` definitions and script-based
language heuristic. The main per-language analysis instead uses canonical tags;
two canonically Mandarin test records fell into the heuristic's English group
(1,002/998 versus canonical 1,000/1,000). The original six retrains used matched
2,000-step LoRA configurations. The monolingual comparison remains confounded
by training-set size and should be interpreted directionally.

## Correctness changes and reproduction limits

The imported source-series bootstrap converted repeated cluster draws to a set,
discarding their bootstrap multiplicity. The corrected code repeats each drawn
cluster and aligns paired models on common IDs. Series IDs are parsed from
`original_id`; records without a matching series use an ID-level singleton
fallback, so the reported cluster count must accompany interpretation. The authentic sample bootstrap
now shares ID draws across dimensions, and the speaker bootstrap excludes the
two-clip speaker to match the four-speaker protocol. These corrections can alter
the saved rebuttal uncertainty intervals; regenerate and review them before
substituting any manuscript numbers.

The imported human within-speaker script included the two-clip speaker even
though the model analysis excluded that speaker. Both now use the same eligibility
rule. Pooled human reference numbers also depend on whether they use the full
authentic pool or the clean held-out subset; compare like-for-like populations.

The release-native human-only imputation sensitivity is approximately 0.614 for
realism, distinct from the older 0.625 convention that also floored the model
side. The primary masked value is approximately 0.602. Source provenance must
come from source-specific evidence; the legacy `is_real_audio` flag cannot
reliably establish it. The public dataset pages were placeholders at the release
preparation check, and authorized public subset results may differ from the
full-corpus results. No claim of complete public reproduction is made here.

## Validation

```bash
pip install pytest
python -m pytest -q tests/test_rebuttal_scoring.py
```

Tests cover masking, malformed predictions, continuous scores, trial averaging,
coverage, duplicate IDs, leakage rejection, clean-pool filtering, shared bootstrap
draws, repeated clusters, speaker eligibility, and the command-line JSON output.
See `validation-summary.json` for aggregate verification against saved full-corpus
artifacts. GPU inference and training are not part of these offline checks.
