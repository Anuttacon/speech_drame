# Project Structure Documentation

`run.py` provides the public dataset-to-results entry point; `docs/RUNNING.md` gives
the runnable routes and `requirements/` separates workflow, cloud, local-model, and training dependencies.


The rebuttal update adds `analysis/` (primary masked scoring, saved-experiment analyses, and artifact documentation), `tests/test_rebuttal_scoring.py`, and `finetuning/scripts/make_ablation_splits.py`. Start with [the analysis guide](analysis/README.md) for the revised results and clean authentic-speech protocol.

This document provides a comprehensive overview of the Speech Drama Evaluation Framework project structure.

## 📁 Root Directory Structure

```
speech_drame/
├── README.md                    # Main project documentation
├── PROJECT_STRUCTURE.md         # This file - project structure documentation
├── requirements.txt             # Python dependencies
├── data/                        # Evaluation datasets
├── zeroshot/                    # Zero-shot evaluation framework
├── fewshot/                     # Few-shot evaluation framework
├── finetuning/                  # Fine-tuning framework
└── scripts/                     # Utility scripts and tools
```

## 📊 Data Directory

```
data/
├── archetype_data/              # Character archetype evaluation data
│   ├── test.jsonl              # Test dataset for archetype evaluation
│   └── *.wav                   # Audio files for archetype evaluation
└── realism_data/               # Realism evaluation data
    ├── test.jsonl              # Test dataset for realism evaluation
    └── *.wav                   # Audio files for realism evaluation
```

**Data Format:**
- **JSONL files**: Each line contains a JSON object with evaluation data
- **Audio files**: WAV format audio files for speech evaluation
- **Annotations**: Human-annotated scores for various evaluation dimensions

## 🎯 Zero-shot Evaluation Framework

```
zeroshot/
├── README.md                    # Zero-shot evaluation documentation
├── models/                      # Audio LLM implementations
│   ├── __init__.py
│   ├── base_audiollm.py        # Abstract base class for audio LLMs
│   ├── gemini25_pro.py         # Google Gemini 2.5 Pro implementation
│   ├── gpt4o.py               # OpenAI GPT-4o implementation
│   ├── qwen25_omni.py         # Alibaba Qwen2.5 Omni implementation
│   ├── qwen2_audio.py         # Alibaba Qwen2 Audio implementation
│   ├── kimi_audio.py          # Moonshot Kimi Audio implementation
│   └── kimiaudio_utility/      # Kimi Audio utility modules
├── prompts/                     # Evaluation prompts and schemas
│   ├── all_prompt.py           # General evaluation prompts
│   ├── all_archetype_prompt.py # Character archetype prompts
│   ├── sep_prompt.py           # Separate dimension prompts
│   ├── sep_archetype_prompt.py # Separate archetype prompts
│   └── schema.py               # JSON schemas for structured output
├── pyscripts/                   # Python evaluation scripts
│   ├── eval.py                 # Main evaluation script
│   ├── score.py                # Scoring utilities
│   ├── score_avg.py            # Average score calculation
│   ├── arche_score.py          # Archetype-specific scoring
│   ├── arche_score_avg.py      # Archetype average scoring
│   ├── utils.py                # Utility functions
│   └── checkpoint_utils.py     # Checkpoint management utilities
└── scripts/                     # Shell scripts for batch evaluation
    ├── eval_*.sh               # Model-specific evaluation scripts
    ├── score_analysis*.sh      # Score analysis scripts
    └── eval_with_checkpoints.sh # Checkpoint-based evaluation
```

## 🎯 Few-shot Evaluation Framework

```
fewshot/
├── README.md                    # Few-shot evaluation documentation
├── models/                      # Audio LLM implementations (shared with zeroshot)
│   ├── __init__.py
│   ├── base_audiollm.py        # Abstract base class for audio LLMs
│   ├── gemini25_pro.py         # Google Gemini 2.5 Pro implementation
│   ├── gpt4o.py               # OpenAI GPT-4o implementation
│   ├── qwen25_omni.py         # Alibaba Qwen2.5 Omni implementation
│   ├── qwen2_audio.py         # Alibaba Qwen2 Audio implementation
│   ├── kimi_audio.py          # Moonshot Kimi Audio implementation
│   └── kimiaudio_utility/      # Kimi Audio utility modules
├── prompts/                     # Few-shot evaluation prompts
│   ├── all_prompt.py           # General few-shot prompts
│   ├── all_archetype_prompt.py # Character archetype few-shot prompts
│   ├── sep_prompt.py           # Separate dimension few-shot prompts
│   ├── sep_archetype_prompt.py # Separate archetype few-shot prompts
│   └── schema.py               # JSON schemas for structured output
├── pyscripts/                   # Python evaluation scripts
│   ├── eval.py                 # Main few-shot evaluation script
│   ├── score.py                # Scoring utilities
│   ├── score_avg.py            # Average score calculation
│   ├── arche_score.py          # Archetype-specific scoring
│   ├── arche_score_avg.py      # Archetype average scoring
│   └── utils.py                # Utility functions
└── scripts/                     # Shell scripts for batch evaluation
    ├── eval_*.sh               # Model-specific evaluation scripts
    └── score_analysis*.sh      # Score analysis scripts
```

## 🎯 Fine-tuning Framework

```
finetuning/
├── README.md                    # Fine-tuning documentation
├── src/                         # Source code
│   ├── train_sft.py            # Main SFT training script
│   ├── train_grpo.py           # GRPO training script
│   ├── test_*.py               # Testing scripts for different tasks
│   ├── analyze_role_results.py # Result analysis script
│   ├── dataset/                 # Dataset handling
│   │   ├── __init__.py
│   │   ├── dataset.py          # Audio dataset implementation
│   │   ├── prompt.py           # Prompt generation utilities
│   │   ├── archetype_prompt.py # Archetype-specific prompts
│   │   └── prompt_variations.json # Prompt variations
│   ├── trainer/                 # Custom trainers
│   │   ├── __init__.py
│   │   ├── sft_trainer.py      # SFT trainer implementation
│   │   ├── grpo_trainer.py     # GRPO trainer implementation
│   │   └── grpo_trainer_new.py # Updated GRPO trainer
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── peft_utils.py       # PEFT configuration utilities
│       ├── rewards.py          # Reward computation
│       └── show_acc.py         # Accuracy visualization
├── conf/                        # Configuration files
│   ├── sft_ft.yaml             # SFT fine-tuning configuration
│   ├── sft_lora.yaml           # SFT with LoRA configuration
│   ├── ds_zero1.json           # DeepSpeed ZeRO-1 configuration
│   ├── ds_zero2.json           # DeepSpeed ZeRO-2 configuration
│   └── ds_zero3.json           # DeepSpeed ZeRO-3 configuration
├── scripts/                     # Training and evaluation scripts
│   ├── parse_result_*.py       # Result parsing utilities
│   ├── prompt_generation.py    # Prompt generation scripts
│   └── show_result_folder.py   # Result visualization
└── run_*.sh                     # Shell scripts for training and testing
    ├── run_archetype_train.sh  # Archetype training
    ├── run_archetype_test.sh   # Archetype testing
    ├── run_realism_train.sh    # Realism training
    └── run_realism_test.sh     # Realism testing
```

## 🛠️ Scripts Directory

```
scripts/
├── README.md                    # Scripts documentation
└── format_code.py              # Code formatting utility
```

## 📋 Key Components

### 1. Model Implementations
- **BaseAudioLLM**: Abstract base class defining the interface for audio LLMs
- **Model-specific implementations**: Concrete implementations for different AI models
- **Utility modules**: Supporting code for specific model integrations

### 2. Prompt Engineering
- **General prompts**: Standard evaluation prompts for all dimensions
- **Archetype prompts**: Character-specific evaluation prompts
- **Separate prompts**: Individual dimension evaluation prompts
- **Schema definitions**: JSON schemas for structured model outputs

### 3. Evaluation Scripts
- **Main evaluation**: Core evaluation logic for different approaches
- **Scoring utilities**: Functions for processing and analyzing results
- **Batch processing**: Shell scripts for running evaluations at scale
- **Checkpointing**: Utilities for resuming interrupted evaluations

### 4. Training Framework
- **SFT training**: Supervised fine-tuning implementation
- **GRPO training**: Group Relative Policy Optimization
- **Dataset handling**: Audio-text dataset processing
- **PEFT support**: Parameter-efficient fine-tuning methods
- **Configuration**: YAML and JSON configuration files

### 5. Data Processing
- **Audio handling**: Audio file loading and preprocessing
- **Annotation processing**: Human annotation parsing and validation
- **Format conversion**: Data format standardization
- **Quality control**: Data validation and cleaning

## 🔄 Data Flow

### Evaluation Flow
1. **Data Loading**: Load evaluation datasets and audio files
2. **Prompt Generation**: Create evaluation prompts based on task type
3. **Model Inference**: Run models on audio-text pairs
4. **Result Processing**: Parse and validate model outputs
5. **Scoring**: Calculate evaluation metrics and scores
6. **Analysis**: Generate performance reports and comparisons

### Training Flow
1. **Data Preparation**: Process training datasets
2. **Model Loading**: Load base models and configurations
3. **Training Loop**: Execute training with monitoring
4. **Checkpointing**: Save model states and metrics
5. **Evaluation**: Test trained models on validation data
6. **Deployment**: Package models for inference

## 🎯 Evaluation Dimensions

The framework supports two distinct evaluation approaches with different dimensional rubrics:

### 🎭 **Realism Evaluation (11 Dimensions)**
Evaluates speech drama generation for realistic character portrayal and narrative consistency:

1. **Pitch Dynamics**: Pitch contour variety and appropriateness
2. **Rhythmic Naturalness**: Flow, timing, pause quality
3. **Stress Emphasis**: Syllable/word emphasis clarity
4. **Emotion Accuracy**: Emotion-label matching
5. **Emotion Intensity**: Emotion strength
6. **Emotional Dynamic Range**: Emotion variation
7. **Voice Identity Matching**: Vocal identity consistency
8. **Trait Embodiment**: Character trait presence
9. **Local Scene Fit**: Immediate scene appropriateness
10. **Global Story Fit**: Overall story consistency
11. **Semantic Matchness**: Content-scene goal alignment

### 🎪 **Archetype Evaluation (4 Dimensions)**
Evaluates speech generation for character archetype consistency and role-playing accuracy:

1. **Content Pass**: Binary pass/fail for basic requirements (length, relevance, language)
2. **Audio Quality**: Technical quality assessment (artifacts, glitches)
3. **Human Likeness**: Context-free naturalness of speech delivery
4. **Appropriateness**: Context-dependent fit to role and scene requirements

## 🔧 Configuration Management

### YAML Configuration Files
- **Training parameters**: Learning rates, batch sizes, epochs
- **Model settings**: Model paths, initialization parameters
- **PEFT configuration**: LoRA, QLoRA, and other PEFT methods
- **Logging settings**: TensorBoard, wandb, logging frequency

### JSON Configuration Files
- **DeepSpeed settings**: ZeRO optimization configurations
- **Dataset configurations**: Data loading and preprocessing settings
- **Evaluation settings**: Evaluation parameters and metrics

## 📊 Output Formats

### Evaluation Results
- **JSONL format**: One result per line for easy processing
- **Structured output**: Consistent format across all evaluation methods
- **Metadata**: Model information, timestamps, and configuration details

### Training Outputs
- **Model checkpoints**: Saved model states and weights
- **Training logs**: Loss curves, metrics, and performance data
- **Configuration backups**: Training configuration preservation

## 🚀 Deployment Considerations

### Model Serving
- **API endpoints**: RESTful interfaces for model inference
- **Batch processing**: Efficient handling of multiple requests
- **Resource management**: GPU memory and compute optimization

### Scalability
- **Distributed training**: Multi-GPU and multi-node support
- **Parallel evaluation**: Concurrent model evaluation
- **Caching**: Result caching for repeated evaluations

## 📝 Documentation Standards

### Code Documentation
- **Docstrings**: Comprehensive function and class documentation
- **Type hints**: Python type annotations for better code clarity
- **Comments**: Inline comments for complex logic

### User Documentation
- **README files**: Module-specific documentation
- **API reference**: Detailed function and class documentation
- **Examples**: Usage examples and tutorials

## 🔍 Quality Assurance

### Code Quality
- **Formatting**: Consistent code style with black and isort
- **Linting**: Code quality checks with flake8
- **Type checking**: Static type analysis with mypy

### Testing
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end workflow testing
- **Performance tests**: Benchmarking and profiling

---

This project structure supports a comprehensive evaluation framework for speech drama generation, providing multiple evaluation approaches and extensive configuration options for research and development.
