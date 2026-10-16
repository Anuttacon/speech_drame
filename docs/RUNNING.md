# Run Speech-DRAME from a dataset

The main entry point is `python run.py`. It connects canonical dataset metadata
and reconstructed audio to model inputs, evaluation/training commands, and masked
scores. You do not need the private experiment-directory layout for this path.

```text
Dataset metadata + reconstructed waveforms
                  |
             prepare
                  |
     inputs.jsonl + metadata.jsonl + run.json
          |                         |
        infer                     train
          |                         |
   named predictions        your checkpoint
          |
        score
          |
 scores-<name>.json (coverage, per-dimension/per-language correlations, CIs)
```

## 1. Choose a runnable path

Run from the repository root, using a separate environment for model execution:

```bash
python run.py options
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/workflow.txt
```

| Your starting point | What you can run | Additional requirements |
| --- | --- | --- |
| No dataset, checkpoint, API key, or GPU | `demo` | NumPy/SciPy; generated fixtures only |
| Canonical labels and saved predictions | `prepare --score-only` then `score --predictions` | No waveform/model |
| Metadata plus reconstructed audio | `prepare`, `infer`, `score` | Local GPU model or your cloud account |
| Canonical metadata plus mel features only | Scoring saved predictions immediately; inference after reconstruction | Supply a reconstruction map or a companion snapshot with reconstructed audio |
| Labeled training subset plus audio | `prepare --purpose train`, `train` | GPU training stack and explicit training recipe |
| Your trained DRAME checkpoint | `infer --backend drame` | Full weights or adapter plus its base model |

Check the plumbing immediately:

```bash
python run.py demo --output runs/demo
```

This generates eight tiny tone WAVs, canonical-shaped metadata, and deliberately
fixed predictions, then prepares inputs and scores them. Its score JSON records
`workflow.demo_fixture: true`. These are test fixtures, **not model judgments,
benchmark samples, or evidence of benchmark performance**. No API call, dataset
access, model download, or GPU is needed.

## 2. Obtain a dataset snapshot

Dataset destinations are `Anuttacon/speech-drame-evalbench` (metadata/mels) and
`ftshijt/speech-drame-evalbench` (same authorized IDs plus reconstructions). Use
the revision supplied with the dataset release, preferably a commit hash:

```bash
python run.py fetch \
  --repo ftshijt/speech-drame-evalbench \
  --revision <dataset-commit> \
  --output datasets/ftshijt
```

`fetch` uses the [Hugging Face snapshot downloader](https://huggingface.co/docs/huggingface_hub/en/guides/download)
and prints available `metadata/*.jsonl` files. It reports an error if the snapshot
contains only a README. The pages were placeholders at the October 16, 2026
preparation check; this command cannot create missing data. Check the published
file list before attempting inference. Authentication, if needed, uses your
normal Hugging Face environment; credentials are not written into run files.

`prepare` takes exact metadata file(s) and an asset root. It does not assume an
unpublished final archive layout. In the inspected experimental candidates, each
track has its own release root:

```text
<realism-root>/metadata/{train,test_base,test_realrec,benchmark}.jsonl
<archetype-root>/metadata/{train,test}.jsonl
<track-root>/audio_synth/<split>/*.{flac,wav}
<track-root>/spectrogram/<split>/*.npz
```

If a future snapshot nests tracks under `realism/` and `archetype/`, use those
track directories as the asset roots. Do not rename IDs or substitute the entire
authentic pool for its clean split manifest. Public subset results can differ
from the full-corpus paper values.

## 3. Prepare one evaluation track

Point to the directory relative to which the metadata's audio paths resolve:

```bash
python run.py prepare \
  --track realism \
  --metadata /data/realism/metadata/test_base.jsonl \
  --asset-root /data/realism \
  --output runs/realism-test
```

For archetype:

```bash
python run.py prepare \
  --track archetype \
  --metadata /data/archetype/metadata/test.jsonl \
  --asset-root /data/archetype \
  --output runs/archetype-test
```

Preparation validates perspective, IDs, required context, annotations, and audio
paths. It reads reconstructed waveforms from `audio.synthesized.path`, converts
FLAC/other formats to real PCM WAV files, and writes:

| Prepared file | Purpose |
| --- | --- |
| `metadata.jsonl` | Selected canonical records and masked labels for scoring |
| `inputs.jsonl` | Inputs accepted by baseline and fine-tuning scripts |
| `audio/<id>.wav` | Staged reconstructed audio, never inferred from `is_real_audio` |
| `run.json` | Track, purpose, protocol, counts, source/input/audio hashes, audio readiness |
| `train_ids.jsonl`, `test_ids.jsonl` | Membership evidence when supplied |

The conversion is explicit: realism `profile` -> `char_profile`, `scene.local` ->
`local_scene`, and canonical rater names -> the existing training/inference
annotation names. Archetype `prompt` -> `question`, and `dataset_name` is set to
`Arche`. Canonical scoring metadata is retained separately, so this mapping does
not replace missing human labels with floor scores. The archetype release has a
consensus Content Pass boolean; its model-input representation is a one-element
list, not a recovered individual-rater vote history.

For a mel-only snapshot, reconstruct audio using the pinned frontend/vocoder
recipe accompanying that release, or use the companion reconstruction snapshot.
The complete public reconstruction recipe is still a release dependency; the CLI
does not invent mel settings or interpret features as waveforms. Alternatively,
supply locally reconstructed audio with a JSONL map:

```json
{"id":"your-record-id","wav_path":"/data/reconstructions/your-record-id.wav"}
```

Add `--audio-map reconstructions.jsonl` to `prepare`. Relative mapped paths
resolve against `--asset-root`. To score existing predictions without any audio,
add `--score-only`; model execution rejects a score-only preparation.

## 4. Select and run an evaluator

Each prediction run has its own `--name`, so baseline, few-shot, full-model, and
ablation outputs remain separate. Commands refuse to overwrite an existing run.

### Local baseline

The local requirements record the core versions found in the saved experiment
environment (PyTorch/Torchaudio 2.7.1, Transformers 4.51.0, Accelerate 1.8.1,
PEFT 0.16.0). Use Python 3.10/3.11 and a compatible Linux/CUDA installation for
model runs; install the appropriate PyTorch build for your machine. Model/GPU
execution was not rerun in this PR.

```bash
pip install -r requirements/local.txt
python run.py infer --workdir runs/realism-test \
  --backend qwen2audio --name qwen-zs --dry-run
python run.py infer --workdir runs/realism-test \
  --backend qwen2audio --name qwen-zs
python run.py score --workdir runs/realism-test --name qwen-zs
```

`qwen25` selects the Qwen2.5-Omni adapter. `--model` overrides a baseline's model
ID. Local execution may download the base model. `--dry-run` validates prepared
inputs and prints commands without model loading or API calls; it does not
certify GPU memory, package compatibility, or cloud credentials.

### Cloud baseline

Only the selected backend is imported; cloud runs do not require local model
weights or a GPU:

```bash
pip install -r requirements/cloud.txt
export GEMINI_API_KEY=<your-key>
python run.py infer --workdir runs/realism-test \
  --backend gemini25 --model <available-gemini-model-id> \
  --trials 5 --name gemini-zs
python run.py score --workdir runs/realism-test --name gemini-zs
```

Gemini also accepts `GOOGLE_API_KEY`, or Vertex application default credentials
with `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and optionally
`GOOGLE_GENAI_USE_VERTEXAI=true`, following the [Google SDK configuration](https://googleapis.github.io/python-genai/).
No private project or developer credential path is embedded.

The existing `gpt4o` adapter uses **Azure OpenAI**:

```bash
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_API_ENDPOINT=<your-Azure-endpoint>
python run.py infer --workdir runs/realism-test \
  --backend gpt4o --model <your-audio-enabled-deployment> --name azure-zs
```

Executing cloud inference sends the prepared audio and prompts to your selected
provider and uses that account's quota. Available model IDs and deployments are
account-specific; a current model is not guaranteed to reproduce an older API
snapshot. No paid inference is run by the offline demo or dry-run commands.

### DRAME checkpoint and modality controls

Published trained weights remain a release dependency. Given your own checkpoint:

```bash
python run.py infer --workdir runs/realism-test \
  --backend drame --checkpoint /models/realism/checkpoint-10000 --name drame
python run.py score --workdir runs/realism-test --name drame
```

The command runs all 10 realism dimensions or all three primary archetype score
dimensions automatically. For a local LoRA directory containing
`adapter_config.json`, inference loads the specified base model and attaches the
adapter using [PEFT](https://huggingface.co/docs/peft/main/en/guides/peft_model_config).
Use `--base-model /models/local-base` to override an obsolete base-model path.
For full model weights, use the full checkpoint directory. The realism and
archetype adapters are separate models; choose the corresponding checkpoint.

For realism, append `--zero-audio` or `--zero-text` and use a different `--name`.
These flags correspond to the rebuttal's silence and audio-only controls. Saved
predictions record the flags. This is different from the separately trained
gold-transcript text-only baseline.

### Few-shot baseline

First prepare a **separate training** subset using `--purpose train`. Then:

```bash
python run.py infer --workdir runs/realism-test \
  --backend gemini25 --model <available-gemini-model-id> \
  --examples runs/realism-examples --num-examples 3 --trials 5 --name gemini-fs
python run.py score --workdir runs/realism-test --name gemini-fs
```

The workflow checks the example bundle's track/purpose and rejects overlapping
example/test IDs. Human labels in the example bundle are intentional few-shot
supervision. The existing prompting utility's historical example-label
aggregation remains unchanged; primary *evaluation* uses masking.

## 5. Clean authentic-speech evaluation

Use explicit training and clean test manifests with `id` fields. The full pool
contains both adaptation and held-out records:

```bash
python run.py prepare \
  --track realism --protocol authentic \
  --metadata /data/realism/metadata/test_realrec.jsonl \
  --asset-root /data/realism \
  --train-ids /data/manifests/train_all_real_v2.jsonl \
  --test-ids /data/manifests/test_real_v2.jsonl \
  --output runs/authentic-test
python run.py infer --workdir runs/authentic-test \
  --backend drame --checkpoint /models/realism/checkpoint-10000 --name drame
python run.py score --workdir runs/authentic-test --name drame
```

Preparation rejects overlap and requires selected test IDs to exist in metadata.
It stores the membership evidence used by scoring. Add `--ids authorized-test-ids.jsonl`
to select an exact authorized subset; those IDs must belong to the test manifest.
The score reports actual coverage and within-speaker results, and does not assume
that a public subset still contains all 1,084 experimental test records.

## 6. Train your own evaluator

```bash
pip install -r requirements/training.txt
python run.py prepare \
  --track realism --purpose train \
  --metadata /data/realism/metadata/train.jsonl \
  --asset-root /data/realism --output runs/realism-train
python run.py train --workdir runs/realism-train \
  --config finetuning/conf/sft_lora.yaml --output runs/my-realism-model --dry-run
python run.py train --workdir runs/realism-train \
  --config finetuning/conf/sft_lora.yaml --output runs/my-realism-model
```

The command supplies the prepared input and output paths to the existing trainer;
choose hyperparameters and DeepSpeed settings in the YAML. The existing LoRA
recipe uses DeepSpeed; training requirements include it. Single-process launch is
supported; multi-GPU/cluster scheduling remains outside this wrapper. Training
artifacts are placed under the requested output, and saved checkpoints can feed
back into `infer --backend drame`.

For archetype, use `--track archetype` and its training metadata. For realism
training with authentic adaptation, provide both base-training and authentic-pool
metadata after `--metadata`, plus explicit `--train-ids` and `--test-ids`; the
workflow selects training IDs only and rejects any overlap. Without membership
manifests, training accepts only records explicitly marked `split=train`.

This is a release-native training path, not a claim of bit-identical recovery of
the private original model inputs. The legacy trainer samples individual ratings
and floors empty training-target arrays; it retains that training convention.
Primary masked evaluation is separate. Canonical release fields omit some
original rater history, and an authorized public subset may omit training items.
The [negative/monolingual filters](../analysis/README.md#checkpoints-and-training-controls)
remain available for the corresponding training ablations.

## 7. Read and share results

`score` writes `scores-<name>.json`: per-dimension counts/correlations, the primary
dimension-average correlation, canonical per-language values, bootstrap CIs, ID
coverage, and a workflow manifest hash. Undefined dimensions produce `null` for
the headline average. It keeps source paths/hashes and the prediction-file hashes
so the run can be traced to inputs and chosen commands.

To use existing predictions rather than execute a model:

```bash
python run.py score --workdir runs/realism-test \
  --predictions /saved/trial1.jsonl /saved/trial2.jsonl --output runs/saved-scores.json
```

A prepared bundle is local to its staged absolute audio paths. Prepare again on
another machine. Do not publish a run directory wholesale: it contains dataset
records and reconstructed audio whose release scope follows the dataset, plus
local paths. Share permitted aggregates and the exact dataset/model revisions.
See [the analysis guide](../analysis/README.md) for advanced rebuttal experiments
and [review notes](REBUTTAL_UPDATES.md) for known manuscript-number discrepancies.
