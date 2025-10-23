# Few-shot Speech Drama Evaluation

This module provides few-shot evaluation capabilities for AI models on speech drama generation tasks. It extends the zero-shot framework with example-based prompting to improve model performance through in-context learning.

## 🎯 Overview

Few-shot evaluation tests models with a limited number of examples provided in the prompt, allowing models to learn task-specific patterns through in-context learning without requiring fine-tuning.

## 🏗️ Architecture

```
fewshot/
├── models/           # Audio LLM implementations (shared with zeroshot)
│   ├── base_audiollm.py      # Abstract base class
│   ├── gemini25_pro.py       # Google Gemini 2.5 Pro
│   ├── gpt4o.py             # OpenAI GPT-4o
│   ├── qwen25_omni.py       # Alibaba Qwen2.5 Omni
│   ├── qwen2_audio.py       # Alibaba Qwen2 Audio
│   └── kimi_audio.py        # Moonshot Kimi Audio
├── prompts/          # Few-shot evaluation prompts
│   ├── all_prompt.py        # General few-shot prompts
│   ├── all_archetype_prompt.py  # Character archetype few-shot prompts
│   ├── sep_prompt.py        # Separate dimension few-shot prompts
│   ├── sep_archetype_prompt.py  # Separate archetype few-shot prompts
│   └── schema.py            # JSON schemas for structured output
├── pyscripts/        # Python evaluation scripts
│   ├── eval.py              # Main few-shot evaluation script
│   ├── score.py             # Scoring utilities
│   ├── score_avg.py         # Average score calculation
│   ├── arche_score.py       # Archetype-specific scoring
│   ├── arche_score_avg.py   # Archetype average scoring
│   └── utils.py             # Utility functions
└── scripts/          # Shell scripts for batch evaluation
    ├── eval_*.sh            # Model-specific evaluation scripts
    └── score_analysis*.sh   # Score analysis scripts
```

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**
   ```bash
   conda activate fewshot
   export PYTHONPATH=/path/to/speech_drame/fewshot:${PYTHONPATH}
   ```

2. **API Keys** (for cloud-based models)
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   # Add other API keys as needed
   ```

### Basic Usage

#### Single Model Few-shot Evaluation
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

#### Batch Few-shot Evaluation
```bash
# Evaluate Gemini 2.5 Pro with few-shot examples
bash scripts/eval_gemini.sh

# Evaluate GPT-4o with few-shot examples
bash scripts/eval_gpt4o.sh
```

## 📊 Few-shot Evaluation Types

The few-shot framework supports two distinct evaluation approaches:

### 🎭 **Realism Few-shot Evaluation**
Provides examples for realistic character portrayal and narrative consistency.

#### 1. Combined Realism Few-shot
Provides examples for all 11 dimensions in a single prompt:
```bash
--prompt_type combined
--data_path ../data/realism_data/test.jsonl
```

#### 2. Separate Realism Few-shot
Provides examples for each dimension individually:
```bash
--prompt_type separate
--data_path ../data/realism_data/test.jsonl
```

### 🎪 **Archetype Few-shot Evaluation**
Provides examples for character archetype consistency and role-playing accuracy.

#### 1. Combined Archetype Few-shot
Provides examples for all 4 dimensions in a single prompt:
```bash
--prompt_type archetype_combined
--data_path ../data/archetype_data/test.jsonl
```

#### 2. Separate Archetype Few-shot
Provides examples for each dimension individually:
```bash
--prompt_type archetype_separate
--data_path ../data/archetype_data/test.jsonl
```

## 🎯 Few-shot Prompting Strategy

### Example Selection
The few-shot framework automatically selects high-quality examples based on:
- **Score Quality**: Examples with high human ratings
- **Diversity**: Covering different character types and scenarios
- **Clarity**: Clear, unambiguous examples
- **Balance**: Representing various difficulty levels

### Prompt Structure
```
[System Prompt]
[Evaluation Rubric]
[Few-shot Examples]
[Target Audio + Context]
```

### Example Format
Each few-shot example includes:
- **Audio**: Reference audio file
- **Context**: Character profile and scene description
- **Transcript**: Spoken content
- **Scores**: Human-annotated scores for all dimensions
- **Explanation**: Brief rationale for scores

## 🤖 Supported Models

Same as zero-shot evaluation:

| Model | Implementation | API Type | Few-shot Support |
|-------|---------------|----------|------------------|
| **Gemini 2.5 Pro** | `gemini25_pro.py` | Google Vertex AI | ✅ |
| **GPT-4o** | `gpt4o.py` | OpenAI API | ✅ |
| **Qwen2.5 Omni** | `qwen25_omni.py` | Local/API | ✅ |
| **Qwen2 Audio** | `qwen2_audio.py` | Local/API | ✅ |
| **Kimi Audio** | `kimi_audio.py` | Moonshot API | ✅ |

## 📈 Evaluation Metrics

Uses the same 11-dimensional rubric as zero-shot evaluation:

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

## 🔧 Advanced Features

### Dynamic Example Selection
- **Adaptive Examples**: Select examples based on target character type
- **Difficulty Matching**: Match example difficulty to target complexity
- **Domain Adaptation**: Use examples from similar scenarios

### Few-shot Variants
- **1-shot**: Single example per dimension
- **3-shot**: Three examples per dimension
- **5-shot**: Five examples per dimension
- **Mixed-shot**: Different numbers for different dimensions

### Example Quality Control
- **Human Validation**: All examples manually verified
- **Score Consistency**: Examples with consistent human ratings
- **Diversity Metrics**: Ensure example diversity across dimensions

## 📊 Results Analysis

### Compare Zero-shot vs Few-shot
```bash
# Generate comparison analysis
python pyscripts/compare_zeroshot_fewshot.py \
    --zeroshot_results zeroshot_results.jsonl \
    --fewshot_results fewshot_results.jsonl \
    --output_file comparison_analysis.json
```

### Few-shot Effectiveness Analysis
```bash
# Analyze few-shot improvement
python pyscripts/analyze_fewshot_improvement.py \
    --input_dir results/ \
    --output_file fewshot_improvement.json
```

### Batch Analysis
```bash
# General score analysis
bash scripts/score_analysis.sh

# Archetype score analysis
bash scripts/score_analysis_arche.sh
```

## 📁 Data Format

### Input Data (JSONL)
Same format as zero-shot evaluation:
```json
{
  "id": "sample_001",
  "local_scene": "Character description and scene context",
  "char_age": "Adult",
  "char_style": "Playful, Charismatic, Confident",
  "char_profile": "Detailed character background",
  "transcript": "Spoken text content",
  "wav_path": "path/to/audio/file.wav"
}
```

### Few-shot Example Data
```json
{
  "id": "example_001",
  "local_scene": "Example scene context",
  "char_age": "Teenager",
  "char_style": "Kind-hearted, Compassionate",
  "char_profile": "Example character background",
  "transcript": "Example spoken content",
  "wav_path": "path/to/example/audio.wav",
  "scores": {
    "pitch_dynamics": 4.0,
    "rhythmic_naturalness": 3.0,
    "stress_emphasis": 4.0,
    "emotion_accuracy": 5.0,
    "emotion_intensity": 4.0,
    "emotional_dynamic_range": 3.0,
    "voice_identity_matching": 4.0,
    "trait_embodiment": 5.0,
    "local_scene_fit": 4.0,
    "global_story_fit": 4.0,
    "semantic_matchness": 4.0
  },
  "explanation": "Brief explanation of scores"
}
```

### Output Data (JSONL)
Same format as zero-shot evaluation:
```json
{
  "id": "sample_001",
  "model_name": "gemini25",
  "response": "{\"pitch_dynamics\": 4.0, \"rhythmic_naturalness\": 3.0, ...}",
  "audio_path": "path/to/audio/file.wav",
  "few_shot_examples": ["example_001", "example_002", "example_003"]
}
```

## 🛠️ Configuration

### Few-shot Configuration
- **Example Count**: Number of examples per dimension
- **Example Selection**: Strategy for choosing examples
- **Prompt Format**: How examples are presented
- **Context Matching**: How to match examples to targets

### Model Configuration
Same as zero-shot evaluation with additional few-shot specific settings:
- **Context Length**: Manage prompt length with examples
- **Example Ordering**: Optimal arrangement of examples
- **Token Limits**: Handle model context limitations

## 🔍 Troubleshooting

### Common Issues

1. **Context Length Limits**
   - Reduce number of examples
   - Use shorter example descriptions
   - Implement dynamic example selection

2. **Example Quality Issues**
   - Validate example scores
   - Check example diversity
   - Ensure example relevance

3. **Performance Degradation**
   - Compare with zero-shot baseline
   - Analyze example-target similarity
   - Adjust example selection strategy

### Debug Mode
Enable verbose logging for few-shot evaluation:
```bash
export PYTHONPATH=/path/to/speech_drame/fewshot:${PYTHONPATH}
python -u pyscripts/eval.py [args] 2>&1 | tee fewshot_evaluation.log
```

## 📚 API Reference

### Few-shot Specific Methods
```python
def select_few_shot_examples(target_item: dict, example_pool: list, num_examples: int) -> list:
    """Select optimal few-shot examples for target item"""
    
def format_few_shot_prompt(examples: list, target_item: dict, rubric: str) -> str:
    """Format few-shot prompt with examples and target"""
    
def analyze_few_shot_effectiveness(results: list) -> dict:
    """Analyze effectiveness of few-shot prompting"""
```

### Example Management
- **Example Pool**: Database of validated examples
- **Selection Algorithms**: Various strategies for example selection
- **Quality Metrics**: Measures of example quality and diversity

## 🎓 Best Practices

### Example Selection
1. **Diversity**: Choose examples covering different scenarios
2. **Quality**: Use high-quality, well-annotated examples
3. **Relevance**: Match examples to target characteristics
4. **Balance**: Include both positive and negative examples

### Prompt Engineering
1. **Clear Structure**: Organize examples logically
2. **Consistent Format**: Maintain consistent example format
3. **Appropriate Length**: Balance detail with context limits
4. **Clear Instructions**: Provide explicit evaluation guidance

### Evaluation Strategy
1. **Baseline Comparison**: Always compare with zero-shot
2. **Multiple Trials**: Run multiple evaluations for reliability
3. **Error Analysis**: Analyze cases where few-shot fails
4. **Iterative Improvement**: Refine examples based on results

## 🤝 Contributing

1. **Adding New Example Selection Strategies**
   - Implement selection algorithms
   - Add to example selection logic
   - Validate with experiments

2. **Improving Few-shot Prompts**
   - Design better prompt templates
   - Optimize example presentation
   - Test different prompt structures

3. **Enhancing Analysis Tools**
   - Add few-shot effectiveness metrics
   - Implement comparison tools
   - Create visualization utilities

## 📄 License

This module is part of the Speech Drama Evaluation Framework and follows the same licensing terms.

---

For more detailed information, see the main project README and zero-shot evaluation documentation.
