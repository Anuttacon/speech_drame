# Rebuttal code review and update

Draft PR: https://github.com/Anuttacon/speech_drame/pull/3

Reviewed against the revised manuscript, three reviewer responses, and the saved
rebuttal experiment source. The public starting point is upstream commit
`d05e1bc`, which predates the rebuttal analyses. The original analysis source
hashes are in `analysis/imported-source-sha256.json`; ported code and validation
are described in `analysis/README.md`.

## Implemented corrections

| Finding | Change |
| --- | --- |
| Baseline scorers replace human N/A with score 1, conflating missingness with poor performance. | Added a primary canonical-metadata scorer using per-dimension masking; historical entrypoints warn and direct users to it. |
| Missing model responses silently become floor scores; some baseline parsers truncate continuous scores. | Primary scorer preserves finite continuous scores, excludes malformed predictions, and reports coverage. |
| Flattened correlations can be confused with the paper's dimension average. | Primary scorer reports the mean of all expected dimension correlations, with per-dimension counts and per-language results. Missing/constant dimensions make the headline average null. |
| The authentic pool contains adaptation and held-out items. | Authentic scoring requires explicit training/test manifests, rejects overlap/duplicates, and filters the metadata pool before scoring. Clean experiment helpers validate the manifests too. |
| The shared analysis registry points to a different archetype run. | Registered the verified `addcontent` checkpoint output as the primary archetype run. |
| Rebuttal source uses one developer's absolute root. | Added configurable artifact/output roots and portable direct-file scoring. |
| Source-series bootstrap discards repeated cluster draws by converting them to a set. | Preserve repeated draws and align paired comparisons on common IDs. Both strongest baseline comparisons are available. |
| Authentic sample bootstrap draws independently by dimension. | Use shared ID draws across all dimensions and paired models. |
| Speaker bootstrap and human-reference script include a two-clip speaker despite the stated four-speaker protocol. | Exclude speakers with fewer than three clips from speaker-level analyses, while retaining all 1,084 clips in sample-level evaluation. |
| Modality controls and the DDP fix are missing from public code. | Ported the exact `zero_audio`/`zero_text` controls and non-reentrant gradient checkpointing setting. Added ablation flags to saved outputs. |
| Inference silently replaces missing or unreadable audio with silence. | Both fine-tuning inference entrypoints now raise errors. Intentional silence is available through the explicit realism ablation flag. |
| Training-data filters for the rebuttal are missing. | Added deterministic negative-ablation and transcript-script monolingual split construction, with unknown construction types rejected. |
| README reports obsolete results and underspecified release status. | Updated values, scope, dataset counts, resource status, scoring instructions, and required CLI arguments. |

The PR includes 40 recovered analysis modules covering uncertainty, reliability,
per-language/domain reporting, shortcut controls, provenance, SSL/MOS baselines,
negative and cross-language retraining analyses, and demographic audits. It
excludes the obsolete report assembler and interim bucket-only diagnostic. The
original experiment source retains some historical commentary; the analysis
guide identifies the primary path and supporting artifacts for each analysis.

## Verified full-corpus results

The new primary scorer was executed in memory against the existing saved data
and predictions on dev-va. Only aggregate verification output was returned; no
raw annotations, predictions, audio, checkpoints, or credentials are published.

| Quantity | Recomputed value |
| --- | --- |
| Archetype pooled Avg-3 | 0.628807058629 |
| Realism benchmark Avg-10 | 0.602022892159 |
| Clean authentic Avg-10 | -0.142079661727 |
| Authentic within-speaker model average | +0.029179005885 |
| Matched four-speaker human reference | +0.339822142744 |
| Archetype within-language macro | 0.389434271038 |

The model headline values reproduce exactly. Realism covers all 2,000 canonical
base-test IDs; authentic scoring covers all 1,084 clean IDs with zero training
ID overlap. Archetype metadata covers 1,499 IDs; one additional prediction ID
outside that metadata is excluded and counted explicitly.

## Numerical differences to review with the manuscript

The corrected uncertainty code produces different intervals from the saved
rebuttal implementation. These changes do not alter the primary model point
estimates, and this PR does not edit the manuscript.

- Corrected source-clustered realism advantage over Gemini few-shot:
  **+0.214856**, 95% CI **[+0.169291, +0.246126]**. The analogous advantage over
  GPT-4o audio few-shot is **+0.296650**, CI **[+0.217597, +0.352894]**.
  The common-ID union contains 1,966 items over 1,013 clusters. Series labels
  come from `original_id`, with ID-level singleton fallbacks; this is not
  clustering over 1,013 verified source series.
- Corrected shared-ID sample bootstrap for authentic speech gives model CI
  **[-0.177375, -0.106932]** and paired model-minus-baseline CI
  **[-0.426869, -0.310542]**, rather than the narrower intervals from independent
  dimension resampling.
- The four-speaker clustered authentic comparison uses **1,082 clips** after
  excluding the two-clip speaker. Its model point is **-0.144724**, model CI
  **[-0.295502, +0.063596]**, and paired-difference CI
  **[-0.568364, +0.011873]**. The interval still includes zero. The 1,084-clip
  sample-level point remains **-0.142080**.
- The saved five-trial authentic baseline computes **0.227288643369**, including
  in the original saved E2 aggregate. The manuscript table lists **0.226**.
  The README now reports the verified saved value and identifies this discrepancy.
- The pooled human single-rater reference is **0.515426667657** on the clean
  held-out subset and **0.561201523922** on the entire authentic pool. The matched
  four-speaker within-speaker reference **0.339822** agrees with the manuscript.
  The two pooled references should not be treated as the same population.
- The release-native human-only imputation sensitivity is approximately **0.614**;
  the older **0.625** convention also floored model scores and cannot be recovered
  from release-native predictions alone.

Aggregate results and counts are in `analysis/validation-summary.json`. Existing
paper intervals and rounded baseline entries need reconciliation with these
corrected computations before a final release uses them as reproduction targets.

## Validation and limits

- **22 targeted tests passed** for masking, malformed output, continuous scores,
  averaging, ID integrity, leakage rejection, authentic filtering, bootstrap
  draws, cluster multiplicity, speaker eligibility, missing-audio failures,
  training-data filters, and CLI JSON output.
- All analysis modules import successfully with only NumPy and SciPy installed;
  optional scikit-learn work is imported only by the relevant experiments.
- Changed Python files pass syntax compilation; the complete diff passes
  whitespace/error checks.
- Saved full-corpus primary points and the two corrected 2,000-resample bootstrap
  comparisons were recomputed without inference or training.

GPU inference, training, API model calls, all secondary experiment drivers, and
full-environment installation were not executed in this review. Their inputs
include artifacts that are not part of the public code repository. The six
training ablations' saved-result analysis and deterministic data filters are
included, while site-specific cluster submission is excluded. Full-public
reproduction still depends on authorized data, split manifests, predictions,
checkpoint packaging, and source-specific provenance. The existing tracked
bytecode and incomplete Kimi gitlink packaging are pre-existing repository
issues; this change adds ignore rules but does not remove tracked artifacts.

## Dataset-to-results workflow follow-up

The public entry point `run.py` now connects a dataset snapshot to runnable
options: `fetch`, `prepare`, `infer`, `train`, `score`, and an offline `demo`.
The new `docs/RUNNING.md` leads with these routes and gives their prerequisites,
commands, file handoffs, and expected outputs. It distinguishes saved-prediction
scoring from model inference, mel features from reconstructed waveforms, and
README placeholders from real payloads.

Preparation maps canonical realism/archetype fields into existing model inputs,
stages WAV audio (including FLAC conversion), checks declared reconstruction
checksums, preserves canonical labels, stores input/audio hashes, and enforces
training/test membership. Inference generates and can execute all required
DRAME dimensions, a selected local/cloud baseline, repeated trials, or separate
few-shot examples. Training consumes a separately prepared training bundle.

The connected workflow also required fixing backend startup: baseline scripts
now import only the selected model; Gemini uses the caller's API key or Vertex
configuration; and both fine-tuned inference scripts explicitly attach local
PEFT adapters to their base models. Dependency groups separate the lightweight
workflow, cloud SDKs, local inference, and training. The local/training core pins
were read from the saved experiment environment; a complete fresh GPU training
installation was not performed.

**45 targeted tests pass**, including local end-to-end fixtures, schema mapping for both tracks, audio conversion,
command generation, leakage rejection, few-shot separation, CLI startup without
GPU libraries, and adapter attachment are covered by the expanded tests. Full
canonical-data conversion on dev-va was not executed: automatic approval review
rejected transmitting the new workflow source to that host. No raw records,
model weights, or media are added. A real dataset/checkpoint/API account is
still required for its corresponding model run; the generated demo is explicitly
marked as a fixture and does not claim benchmark performance.
