# Copyright 2025 Jiatong Shi (Anuttacon)

import argparse
import json
import random
from typing import List, Dict, Any, Optional

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_few_shot_examples(examples_path: str, prompt_type: str) -> List[Dict[str, Any]]:
    """Load few-shot examples from JSONL file"""
    examples = []
    with open(examples_path, "r") as f:
        for line in f:
            info = json.loads(line.strip())
            if "archetype" in prompt_type:
                info = archetype_filter(info)
            else:
                info = realism_filter(info)
            examples.append(info)
    return examples


def archetype_filter(info: Dict[str, Any]) -> Dict[str, Any]:
    annotation_data = info["annotations"]
    new_annotation_info = {}
    available_keys = ["audio_quality", "human_likeness", "appropriateness", "content_pass"]
    for key in available_keys:
        value_list = annotation_data[key]
        average_value = safe_average_value(value_list)
        new_annotation_info[key] = average_value
    info["annotation_prompt"] = json.dumps(new_annotation_info)
    return info

def realism_filter(info: Dict[str, Any]) -> Dict[str, Any]:
    annotation_data = info["annotations"]
    new_annotation_info = {}
    available_keys = ["pitch_variation", "rhythmic_naturalness", "stress_and_emphasis", "emotion_accuracy", "emotion_intensity_control", "dynamic_range", "voice_identity_matching", "trait_embodiment", "local_story_fit", "global_story_coherence", "semantic_match"]
    name_mapping = {
        "pitch_variation": "pitch_dynamics",
        "rhythmic_naturalness": "rhythmic_naturalness",
        "stress_and_emphasis": "stress_emphasis",
        "emotion_accuracy": "emotion_accuracy",
        "emotion_intensity_control": "emotion_intensity",
        "dynamic_range": "emotional_dynamic_range",
        "voice_identity_matching": "voice_identity_matching",
        "trait_embodiment": "trait_embodiment",
        "local_story_fit": "local_scene_fit",
        "global_story_coherence": "global_story_fit",
        "semantic_match": "semantic_matchness"
    }
    for key in available_keys:
        if key not in annotation_data:
            continue
        value_list = annotation_data[key]
        average_value = safe_average_value(value_list)
        new_annotation_info[name_mapping[key]] = average_value
    info["annotation_prompt"] = json.dumps(new_annotation_info)
    return info


def safe_average_value(value_list: List[Any]) -> int:
    num_samples = len(value_list)
    value_sum = 0
    for sample in value_list:
        if sample == "N/A":
            value_sum += 1
        elif sample is None:
            value_sum += 1
        elif type(sample) is bool and sample is False:
            value_sum += 0
        elif type(sample) is bool and sample is True:
            value_sum += 1
        else:
            value_sum += sample
    if num_samples == 0:
        return 1
    return int(round(value_sum / num_samples))


def select_few_shot_examples(
    examples: List[Dict[str, Any]], 
    current_item: Dict[str, Any], 
    num_examples: int, 
    strategy: str = "random"
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples based on the specified strategy
    
    Args:
        examples: List of all available examples
        current_item: The current item being evaluated (for similarity-based selection)
        num_examples: Number of examples to select
        strategy: Selection strategy ("random", "similar", "diverse")
    
    Returns:
        List of selected examples
    """
    if len(examples) <= num_examples:
        return examples
    
    if strategy == "random":
        return random.sample(examples, num_examples)
    
    elif strategy == "similar":
        # For now, implement simple similarity based on character profile overlap
        # This can be enhanced with more sophisticated similarity metrics
        current_profile = current_item.get("char_profile", "").lower()
        current_traits = current_item.get("char_style", "").lower()
        
        # Score examples based on profile and trait similarity
        scored_examples = []
        for example in examples:
            example_profile = example.get("char_profile", "").lower()
            example_traits = example.get("char_style", "").lower()
            
            # Simple word overlap scoring
            profile_overlap = len(set(current_profile.split()) & set(example_profile.split()))
            trait_overlap = len(set(current_traits.split()) & set(example_traits.split()))
            score = profile_overlap + trait_overlap
            
            scored_examples.append((score, example))
        
        # Sort by score (descending) and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:num_examples]]
    
    elif strategy == "diverse":
        # Select examples that are diverse from each other
        # Simple implementation: select examples with different character profiles
        selected = []
        used_profiles = set()
        
        for example in examples:
            profile_key = example.get("char_profile", "")[:50]  # Use first 50 chars as key
            if profile_key not in used_profiles:
                selected.append(example)
                used_profiles.add(profile_key)
                if len(selected) >= num_examples:
                    break
        
        # If we don't have enough diverse examples, fill with random ones
        if len(selected) < num_examples:
            remaining = [ex for ex in examples if ex not in selected]
            selected.extend(random.sample(remaining, min(num_examples - len(selected), len(remaining))))
        
        return selected
    
    else:
        raise ValueError(f"Unknown few-shot strategy: {strategy}")

def wav_to_base64(wav_path: str) -> str:
    """Convert WAV file to base64 string"""
    import base64
    with open(wav_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')