# Zero-shot Speech Drame Evaluation

Start with [the dataset-to-results workflow](../docs/RUNNING.md):
`python run.py prepare` converts the release schema; `infer`, `train`, and `score` connect the steps.
The manual commands below document the underlying scripts.


> Rebuttal update: use [the primary masked scorer](../analysis/README.md) for revised-paper results. The older plotting/scoring commands below retain historical imputation conventions. The clean authentic-speech protocol requires explicit disjoint training/test manifests.

This module provides zero-shot evaluation capabilities for AI models on speech drama generation tasks. It supports multiple state-of-the-art audio language models and comprehensive evaluation metrics.

## 🎯 Overview

Zero-shot evaluation tests models without any task-specific training, providing a baseline assessment of their natural capabilities for speech drama generation and evaluation.

## 🏗️ Architecture

```
zeroshot/
├── models/           # Audio LLM implementations
│   ├── base_audiollm.py      # Abstract base class
│   ├── gemini25_pro.py       # Google Gemini 2.5 Pro
│   ├── gpt4o.py             # OpenAI GPT-4o
│   ├── qwen25_omni.py       # Alibaba Qwen2.5 Omni
│   ├── qwen2_audio.py       # Alibaba Qwen2 Audio
│   └── kimi_audio.py        # Moonshot Kimi Audio
├── prompts/          # Evaluation prompts and schemas
│   ├── all_prompt.py        # General evaluation prompts
│   ├── all_archetype_prompt.py  # Character archetype prompts
│   ├── sep_prompt.py        # Separate dimension prompts
│   ├── sep_archetype_prompt.py  # Separate archetype prompts
│   └── schema.py            # JSON schemas for structured output
├── pyscripts/        # Python evaluation scripts
│   ├── eval.py              # Main evaluation script
│   ├── score.py             # Scoring utilities
│   ├── score_avg.py         # Average score calculation
│   ├── arche_score.py       # Archetype-specific scoring
│   ├── arche_score_avg.py   # Archetype average scoring
│   ├── utils.py             # Utility functions
│   └── checkpoint_utils.py  # Checkpoint management
└── scripts/          # Shell scripts for batch evaluation
    ├── eval_*.sh            # Model-specific evaluation scripts
    ├── score_analysis*.sh   # Score analysis scripts
    └── eval_with_checkpoints.sh  # Checkpoint-based evaluation
```

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**
   ```bash
   conda activate zeroshot
   export PYTHONPATH=/path/to/speech_drame/zeroshot:${PYTHONPATH}
   ```

2. **API Keys** (for cloud-based models)
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   # Add other API keys as needed
   ```

### Basic Usage

#### Single Model Evaluation
```bash
python pyscripts/eval.py \
    --model_name gemini25 \
    --model_tag gemini-2.5-pro \
    --data_path ../data/realism_data/test.jsonl \
    --output_path results.jsonl \
    --prompt_type combined \
    --prompt_version v2 \
    --batch_size 4 \
    --num_workers 4
```

#### Batch Evaluation with Scripts
```bash
# Evaluate Gemini 2.5 Pro
bash scripts/eval_gemini.sh

# Evaluate GPT-4o
bash scripts/eval_gpt.sh

# Evaluate Qwen2.5 Omni
bash scripts/eval_qwen25.sh
```

## 📊 Evaluation Types

The zero-shot framework supports two distinct evaluation approaches:

### 🎭 **Realism Evaluation**
Evaluates speech drama generation for realistic character portrayal and narrative consistency.

#### 1. Combined Realism Evaluation
Evaluates all 11 dimensions in a single prompt:
```bash
--prompt_type combined
--data_path ../data/realism_data/test.jsonl
```

#### 2. Separate Realism Evaluation
Evaluates each dimension individually:
```bash
--prompt_type separate
--data_path ../data/realism_data/test.jsonl
```

### 🎪 **Archetype Evaluation**
Evaluates speech generation for character archetype consistency and role-playing accuracy.

#### 1. Combined Archetype Evaluation
Evaluates all 4 dimensions in a single prompt:
```bash
--prompt_type archetype_combined
--data_path ../data/archetype_data/test.jsonl
```

#### 2. Separate Archetype Evaluation
Evaluates each dimension individually:
```bash
--prompt_type archetype_separate
--data_path ../data/archetype_data/test.jsonl
```

## 🤖 Supported Models

| Model | Implementation | API Type | Schema Support |
|-------|---------------|----------|----------------|
| **Gemini 2.5 Pro** | `gemini25_pro.py` | Google Vertex AI | ✅ |
| **GPT-4o** | `gpt4o.py` | OpenAI API | ✅ |
| **Qwen2.5 Omni** | `qwen25_omni.py` | Local/API | ✅ |
| **Qwen2 Audio** | `qwen2_audio.py` | Local/API | ✅ |
| **Kimi Audio** | `kimi_audio.py` | Moonshot API | ✅ |

## 📈 Evaluation Metrics

### 🎭 **Realism Evaluation Metrics (11 Dimensions)**
1. **Pitch Dynamics** - Pitch contour variety and appropriateness
2. **Rhythmic Naturalness** - Flow, timing, pause quality
3. **Stress Emphasis** - Syllable/word emphasis clarity
4. **Emotion Accuracy** - Emotion-label matching
5. **Emotion Intensity** - Emotion strength
6. **Emotional Dynamic Range** - Emotion variation
7. **Voice Identity Matching** - Vocal identity consistency
8. **Trait Embodiment** - Character trait presence
9. **Local Scene Fit** - Immediate scene appropriateness
10. **Global Story Fit** - Overall story consistency
11. **Semantic Matchness** - Content-scene goal alignment

### 🎪 **Archetype Evaluation Metrics (4 Dimensions)**
1. **Content Pass** - Binary pass/fail for basic requirements (length, relevance, language)
2. **Audio Quality** - Technical quality assessment (artifacts, glitches)
3. **Human Likeness** - Context-free naturalness of speech delivery
4. **Appropriateness** - Context-dependent fit to role and scene requirements

**Scoring Scale:**
- **Realism**: All dimensions scored 1-5 (1 = poor, 5 = excellent)
- **Archetype**: Content Pass (true/false), others scored 1-5

## 🔧 Advanced Features

### Checkpointing
Resume interrupted evaluations:
```bash
python pyscripts/eval.py \
    --checkpoint_dir checkpoints/ \
    --checkpoint_interval 10 \
    --resume_from checkpoints/checkpoint_file.json
```

### Schema Validation
Use structured output for better reliability:
```bash
--use_schema true
```

### Content Pass Mode
For models supporting binary classification:
```bash
--prompt_type archetype_separate  # Automatically detects content_pass
```

## 📊 Results Analysis

### Generate Average Scores
```bash
python pyscripts/score_avg.py \
    --input_dir results/ \
    --output_file analysis.json
```

### Archetype-Specific Analysis
```bash
python pyscripts/arche_score.py \
    --input_dir results/ \
    --output_file archetype_analysis.json
```

### Batch Analysis with Scripts
```bash
# General score analysis
bash scripts/score_analysis.sh

# Archetype score analysis
bash scripts/score_analysis_arche.sh

# Individual score analysis
bash scripts/score_individual.sh
```

## 📁 Data Formats

### 🎭 **Realism Evaluation Data**
**Input Format:**
```json
{
  "id": "role-eval_v1_0000",
  "local_scene": "Character description and scene context",
  "char_age": "Adult",
  "char_style": "Playful, Charismatic, Confident",
  "char_profile": "Detailed character background",
  "transcript": "Spoken text content",
  "wav_path": "path/to/audio/file.wav",
  "annotations": {
    "pitch_variation": [5, 3, 4, 5, 5, 5],
    "rhythmic_naturalness": [5, 4, 4, 5, 2, 5],
    "emotion_accuracy": [4, 4, 4, 5, 5, 4],
    "trait_embodiment": [5, 5, 5, 4, 5, 5]
  }
}
```

**Output Format:**
```json
{
  "id": "role-eval_v1_0000",
  "model_name": "gemini25",
  "response": "{\"pitch_dynamics\": 4.0, \"rhythmic_naturalness\": 3.0, \"emotion_accuracy\": 4.0, ...}",
  "audio_path": "path/to/audio/file.wav"
}
```

### 🎪 **Archetype Evaluation Data**
**Input Format:**
```json
{
  "id": "250814_zh_SocialIdentity_0051",
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

**Output Format:**
```json
{
  "id": "250814_zh_SocialIdentity_0051",
  "model_name": "gemini25",
  "response": "{\"content_pass\": true, \"audio_quality\": 4, \"human_likeness\": 3, \"appropriateness\": 4}",
  "audio_path": "path/to/audio/file.wav"
}
```

## 🛠️ Configuration

### Model Configuration
Each model implementation supports:
- Custom system prompts
- Retry logic with exponential backoff
- Batch processing with thread pools
- Schema-based structured output

### Evaluation Configuration
- **Batch Size**: Control memory usage and API rate limits
- **Num Workers**: Parallel processing threads
- **Prompt Version**: v1 (basic) or v2 (detailed)
- **Checkpoint Interval**: Save progress frequency

## 🔍 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Reduce `batch_size` and `num_workers`
   - Implement longer delays between requests

2. **Memory Issues**
   - Process smaller batches
   - Use checkpointing for large datasets

3. **Schema Validation Errors**
   - Check model compatibility with `--use_schema`
   - Fall back to regular generation if needed

### Debug Mode
Enable verbose logging:
```bash
export PYTHONPATH=/path/to/speech_drame/zeroshot:${PYTHONPATH}
python -u pyscripts/eval.py [args] 2>&1 | tee evaluation.log
```

## 📚 API Reference

### BaseAudioLLM
Abstract base class for all audio LLM implementations:
```python
class BaseAudioLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, audio_path: str) -> Dict[str, Any]
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], audio_paths: List[str]) -> List[Dict[str, Any]]
```

### Model-Specific Methods
- `evaluate_with_schema()`: Schema-based evaluation
- `_ask_with_retry()`: Retry logic with backoff
- Custom prompt handling and response parsing

## 🤝 Contributing

1. **Adding New Models**
   - Inherit from `BaseAudioLLM`
   - Implement required methods
   - Add to model selection logic in `eval.py`

2. **Adding New Prompts**
   - Create prompt functions in appropriate module
   - Add to prompt selection logic
   - Update schema definitions if needed

3. **Improving Evaluation**
   - Enhance scoring algorithms
   - Add new evaluation dimensions
   - Improve error handling and robustness

## 📄 License

This module is part of the Speech Drama Evaluation Framework and follows the same licensing terms.

---

For more detailed information, see the main project README and individual model documentation.
