# Fine-tuning Speech Drama Evaluation

Start with [the dataset-to-results workflow](../docs/RUNNING.md):
`python run.py prepare` converts the release schema; `infer`, `train`, and `score` connect the steps.
The manual commands below document the underlying scripts.


> Rebuttal update: use [the primary masked scorer](../analysis/README.md) for revised-paper results. The older plotting/scoring commands below retain historical imputation conventions. The clean authentic-speech protocol requires explicit disjoint training/test manifests.
> Realism inference now supports the rebuttal `--zero_audio true` and `--zero_text true` controls. Missing or unreadable audio raises an error; intentional silence requires the explicit ablation flag. See [checkpoint and experiment details](../analysis/README.md#checkpoints-and-training-controls).

This module provides fine-tuning capabilities for training AI models on speech drama evaluation tasks. It supports supervised fine-tuning (SFT) with various parameter-efficient fine-tuning (PEFT) methods and comprehensive training configurations.

## 🎯 Overview

Fine-tuning allows models to learn task-specific patterns for speech drama evaluation through supervised training on annotated datasets. This approach typically achieves the best performance compared to zero-shot and few-shot methods.

## 🏗️ Architecture

```
finetuning/
├── src/              # Source code
│   ├── train_sft.py         # Main SFT training script
│   ├── train_grpo.py        # GRPO training script
│   ├── test_*.py            # Testing scripts for different tasks
│   ├── dataset/             # Dataset handling
│   │   ├── dataset.py       # Audio dataset implementation
│   │   ├── prompt.py        # Prompt generation utilities
│   │   ├── archetype_prompt.py  # Archetype-specific prompts
│   │   └── prompt_variations.json  # Prompt variations
│   ├── trainer/             # Custom trainers
│   │   ├── sft_trainer.py   # SFT trainer implementation
│   │   ├── grpo_trainer.py  # GRPO trainer implementation
│   │   └── grpo_trainer_new.py  # Updated GRPO trainer
│   └── utils/               # Utility modules
│       ├── peft_utils.py    # PEFT configuration utilities
│       ├── rewards.py       # Reward computation
│       └── show_acc.py      # Accuracy visualization
├── conf/             # Configuration files
│   ├── sft_ft.yaml          # SFT fine-tuning configuration
│   ├── sft_lora.yaml        # SFT with LoRA configuration
│   ├── ds_zero1.json        # DeepSpeed ZeRO-1 configuration
│   ├── ds_zero2.json        # DeepSpeed ZeRO-2 configuration
│   └── ds_zero3.json        # DeepSpeed ZeRO-3 configuration
├── scripts/          # Training and evaluation scripts
│   ├── parse_result_*.py    # Result parsing utilities
│   ├── prompt_generation.py # Prompt generation scripts
│   └── show_result_folder.py # Result visualization
└── run_*.sh          # Shell scripts for training and testing
    ├── run_archetype_train.sh  # Archetype training
    ├── run_archetype_test.sh   # Archetype testing
    ├── run_realism_train.sh    # Realism training
    └── run_realism_test.sh     # Realism testing
```

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**
   ```bash
   conda activate finetuning
   pip install -r requirements.txt
   ```

2. **GPU Requirements**
   - CUDA-compatible GPU (recommended: 24GB+ VRAM for full fine-tuning)
   - Multiple GPUs for distributed training

3. **Data Preparation**
   - Prepare training data in JSONL format
   - Ensure audio files are accessible

### Basic Usage

#### Supervised Fine-tuning (SFT)
```bash
python src/train_sft.py \
    --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --data_file data/train_data.jsonl \
    --output_dir exp/sft_model \
    --yaml_config conf/sft_ft.yaml
```

#### SFT with LoRA
```bash
python src/train_sft.py \
    --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --data_file data/train_data.jsonl \
    --output_dir exp/sft_lora_model \
    --yaml_config conf/sft_lora.yaml
```

#### Training with Shell Scripts
```bash
# Train on realism data
bash run_realism_train.sh

# Train on archetype data
bash run_archetype_train.sh
```

## 📊 Training Methods

### 1. Supervised Fine-tuning (SFT)
Standard fine-tuning approach:
- **Full Fine-tuning**: Updates all model parameters
- **LoRA**: Low-rank adaptation for parameter efficiency
- **QLoRA**: Quantized LoRA for memory efficiency
- **AdaLoRA**: Adaptive LoRA with dynamic rank allocation

### 2. GRPO (Group Relative Policy Optimization)
Advanced training method for preference learning:
- **Reward-based Training**: Uses human preference data
- **Group-based Optimization**: Optimizes relative preferences
- **Advanced Metrics**: Tracks preference alignment

### 3. Dataset Types
- **ROLE**: Realism evaluation for character portrayal and narrative consistency
- **Archetype**: Character archetype evaluation for role-playing accuracy
- **AVQA**: Audio-visual question answering

## 🤖 Supported Models

| Model | Base Model | Fine-tuning Support | PEFT Support |
|-------|------------|-------------------|--------------|
| **Qwen2-Audio-7B-Instruct** | Qwen2-Audio | ✅ Full | ✅ LoRA/QLoRA |
| **Qwen2-Audio-14B-Instruct** | Qwen2-Audio | ✅ Full | ✅ LoRA/QLoRA |
| **Custom Models** | Any CausalLM | ✅ Full | ✅ All PEFT methods |

## 📈 Training Configurations

### SFT Configuration (sft_ft.yaml)
```yaml
training:
  model_name_or_path: "Qwen/Qwen2-Audio-7B-Instruct"
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.00005
  weight_decay: 0.01
  warmup_steps: 100
  bf16: true
  gradient_checkpointing: true

peft:
  enabled: false
  method: "lora"
  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.1
```

### LoRA Configuration (sft_lora.yaml)
```yaml
training:
  model_name_or_path: "Qwen/Qwen2-Audio-7B-Instruct"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 0.0001

peft:
  enabled: true
  method: "lora"
  lora:
    r: 32
    lora_alpha: 64
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.1
```

## 🔧 Advanced Features

### DeepSpeed Integration
Support for ZeRO optimization:
- **ZeRO-1**: Optimizer state partitioning
- **ZeRO-2**: Gradient partitioning
- **ZeRO-3**: Parameter partitioning

### TensorBoard Logging
Comprehensive training monitoring:
- **Loss Tracking**: Training and validation loss
- **Accuracy Metrics**: Per-score accuracy tracking
- **Learning Rate**: Learning rate scheduling
- **Custom Metrics**: Task-specific metrics

### PEFT Methods
Parameter-efficient fine-tuning options:
- **LoRA**: Low-rank adaptation
- **QLoRA**: Quantized LoRA
- **AdaLoRA**: Adaptive LoRA
- **IA3**: Infused Adapter by Inhibiting and Amplifying Inner Activations
- **Prefix Tuning**: Prefix-based adaptation
- **Prompt Tuning**: Soft prompt learning

## 📁 Data Formats

### 🎭 **Realism Training Data (JSONL)**
```json
{
  "id": "role-eval_v1_0000",
  "dataset_name": "ROLE",
  "local_scene": "Character description and scene context",
  "char_age": "Adult",
  "char_style": "Playful, Charismatic, Confident",
  "char_profile": "Detailed character background",
  "transcript": "Spoken text content",
  "wav_path": "path/to/audio/file.wav",
  "annotations": {
    "pitch_variation": [5, 3, 4, 5, 5, 5],
    "rhythmic_naturalness": [5, 4, 4, 5, 2, 5],
    "stress_and_emphasis": [5, 4, 5, 5, 2, 5],
    "emotion_accuracy": [4, 4, 4, 5, 5, 4],
    "emotion_intensity_control": [5, 4, 4, 5, 5, 5],
    "dynamic_range": [5, 4, 5, 5, 4, 5],
    "voice_identity_matching": [5, 5, 5, 5, 5, 5],
    "trait_embodiment": [5, 5, 5, 4, 5, 5],
    "local_story_fit": [5, 4, 4, 5, 5, 5],
    "global_story_coherence": [5, 4, 4, 5, 5, 5]
  }
}
```

### 🎪 **Archetype Training Data (JSONL)**
```json
{
  "id": "250814_zh_SocialIdentity_0051",
  "dataset_name": "Arche",
  "question": "你是拖延症患者。朋友询问你之前答应帮忙做的一件事进展如何。你说：",
  "wav_path": "path/to/audio/file.wav",
  "annotations": {
    "content_pass": [true, true, true, true, true, true],
    "audio_quality": [4, 5, 4, 4, 4, 4],
    "human_likeness": [3, 4, 3, 4, 4, 4],
    "appropriateness": [4, 4, 4, 4, 5, 5]
  }
}
```

### Processed Training Data
The dataset processor automatically converts raw data to training format:
```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio_url": "path/to/audio.wav"},
        {"type": "text", "text": "Evaluation prompt..."}
      ]
    }
  ],
  "solution": "<answer>4</answer>",
  "audio": [audio_array_data]
}
```

## 🛠️ Configuration

### Training Arguments
- **Model Settings**: Model path, initialization kwargs
- **Data Settings**: Dataset path, batch size, workers
- **Optimization**: Learning rate, weight decay, scheduler
- **Logging**: TensorBoard, wandb, logging frequency
- **Checkpointing**: Save frequency, model saving

### PEFT Configuration
- **Method Selection**: Choose PEFT method
- **Target Modules**: Specify modules to adapt
- **Rank Settings**: LoRA rank and alpha values
- **Dropout**: Regularization settings

### DeepSpeed Configuration
- **ZeRO Stage**: Choose optimization level
- **Offload Settings**: CPU offloading options
- **Memory Optimization**: Gradient and parameter offloading

## 📊 Evaluation and Testing

### Model Testing
```bash
# Test on realism data
python src/test_role.py \
    --model_path exp/sft_model \
    --data_file data/test_realism.jsonl \
    --output_file results.jsonl

# Test on archetype data
python src/test_arche.py \
    --model_path exp/sft_model \
    --data_file data/test_archetype.jsonl \
    --output_file results.jsonl
```

### Result Analysis
```bash
# Parse and analyze results
python scripts/parse_result_folder.py \
    --input_dir results/ \
    --output_file analysis.json

# Visualize results
python scripts/show_result_folder.py \
    --input_dir results/
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO
   - Switch to LoRA/QLoRA

2. **Training Instability**
   - Reduce learning rate
   - Increase warmup steps
   - Check data quality
   - Monitor gradient norms

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data preprocessing
   - Validate evaluation metrics

### Debug Mode
Enable detailed logging:
```bash
export CUDA_LAUNCH_BLOCKING=1
python -u src/train_sft.py [args] 2>&1 | tee training.log
```

## 📚 API Reference

### SFTTrainer
Main training class for supervised fine-tuning:
```python
class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: SFTConfig = None,
        train_dataset: Optional[Dataset] = None,
        peft_config: Optional[PeftConfig] = None,
        enable_tensorboard: bool = True
    )
```

### AudioDataset
Custom dataset for audio-text pairs:
```python
class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, is_perturb=False)
    def __getitem__(self, index) -> dict
```

### PEFT Utilities
```python
def create_peft_config(config: dict) -> PeftConfig:
    """Create PEFT configuration from YAML config"""
```

## 🎓 Best Practices

### Training Strategy
1. **Start Small**: Begin with LoRA before full fine-tuning
2. **Monitor Metrics**: Use TensorBoard for training monitoring
3. **Validate Early**: Check performance on validation set
4. **Save Checkpoints**: Regular checkpointing for recovery

### Data Quality
1. **Clean Data**: Ensure high-quality annotations
2. **Balanced Dataset**: Maintain diversity in training data
3. **Audio Quality**: Use high-quality audio files
4. **Consistent Format**: Standardize data format

### Hyperparameter Tuning
1. **Learning Rate**: Start with 5e-5, adjust based on performance
2. **Batch Size**: Balance memory usage and training stability
3. **Epochs**: Monitor for overfitting, typically 2-5 epochs
4. **Warmup**: Use 10% of total steps for warmup

## 🤝 Contributing

1. **Adding New Models**
   - Implement model loading logic
   - Add to model selection in training scripts
   - Test with different configurations

2. **Adding New PEFT Methods**
   - Implement PEFT configuration
   - Add to PEFT utilities
   - Validate with experiments

3. **Improving Training**
   - Enhance loss computation
   - Add new metrics
   - Optimize data loading

## 📄 License

This module is part of the Speech Drama Evaluation Framework and follows the same licensing terms.

---

For more detailed information, see the main project README and individual training script documentation.
