import json
import logging
import random

import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

prompt_variations_path = Path(__file__).parent / "prompt_variations.json"
with open(prompt_variations_path) as f:
    prompt_variations = json.load(f)

from .prompt import detailed_dimension_prompts, general_prompt, general_prompt_detailed, caption_prompt, emotion_prompt
from .archetype_prompt import detailed_archetype_dimension_prompts, general_archetype_prompt, general_archetype_prompt_detailed, archetype_caption_prompt, archetype_emotion_prompt

def _handle_wav(wav_path, target_rate=16000):
    """
    handle one wav file.
    Return:
        waveform: numpy narray(1d)
    """
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Check if waveform is empty or has invalid dimensions
        if waveform.numel() == 0:
            logging.warning(f"Empty audio file: {wav_path}")
            # Return a small silence tensor (0.1 seconds at target rate)
            return torch.zeros(int(0.5 * target_rate))
        
        # Check if waveform has valid shape (should be [channels, samples])
        if len(waveform.shape) != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
            logging.warning(f"Invalid waveform shape {waveform.shape} for file: {wav_path}")
            # Return a small silence tensor
            return torch.zeros(int(0.5 * target_rate))
        
        # Resample if necessary
        if sample_rate != target_rate:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
                waveform = resampler(waveform)
            except Exception as e:
                logging.warning(f"Resampling failed for {wav_path}: {e}")
                # If resampling fails, return original waveform or silence
                if waveform.shape[1] > 0:
                    # Take first channel and pad/truncate to reasonable length
                    audio = waveform[0]
                    target_length = int(0.5 * target_rate)  # 0.1 seconds
                    if len(audio) > target_length:
                        audio = audio[:target_length]
                    else:
                        audio = torch.cat([audio, torch.zeros(target_length - len(audio))])
                    return audio
                else:
                    return torch.zeros(int(0.5 * target_rate))
        
        # Extract first channel
        audio = waveform[0]
        
        # Final safety check
        if audio.numel() == 0:
            logging.warning(f"Empty audio after processing: {wav_path}")
            return torch.zeros(int(0.5 * target_rate))
            
        return audio
        
    except Exception as e:
        logging.error(f"Failed to load audio file {wav_path}: {e}")
        # Return a small silence tensor as fallback
        return torch.zeros(int(0.5 * target_rate))


def _handle_avqa(obj_avqa):
    choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
    question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the final answer in <answer> </answer>."
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
    # question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_avqa["audio_path"]}, {"type": "text", "text": question_template}]}]
    answer_str = obj_avqa["multi_choice"][obj_avqa["answer"]]
    obj_avqa["solution"] = f"<answer>{answer_str}</answer>"
    return obj_avqa

# Mapping from dimension names to JSON field names
DIMENSION_TO_FIELD_MAPPING = {
    "pitch_dynamics": "pitch_variation",
    "rhythmic_naturalness": "rhythmic_naturalness", 
    "stress_emphasis": "stress_and_emphasis",
    "emotion_accuracy": "emotion_accuracy",
    "emotion_intensity": "emotion_intensity_control",
    "emotional_dynamic_range": "dynamic_range",
    "voice_identity_matching": "voice_identity_matching",
    "trait_embodiment": "trait_embodiment",
    "local_scene_fit": "local_story_fit",
    "global_story_fit": "global_story_coherence"
}

DIMENSION_TO_FIELD_MAPPING_SCHEMA = {
    "pitch_dynamics": "PitchDynamics",
    "rhythmic_naturalness": "RhythmicNaturalness",
    "stress_emphasis": "StressEmphasis",
    "emotion_accuracy": "EmotionAccuracy",
    "emotion_intensity": "EmotionIntensity",
    "emotional_dynamic_range": "EmotionalDynamicRange",
    "voice_identity_matching": "VoiceIdentityMatching",
    "trait_embodiment": "TraitEmbodiment",
    "local_scene_fit": "LocalSceneFit",
    "global_story_fit": "GlobalStoryFit"
}

FIXED_ORDER = [
    "pitch_dynamics",
    "rhythmic_naturalness",
    "stress_emphasis",
    "emotion_accuracy",
    "emotion_intensity",
    "emotional_dynamic_range",
    "voice_identity_matching",
    "trait_embodiment",
    "local_scene_fit",
    "global_story_fit"
]

def get_user_prompt(user_data: dict) -> tuple[str, str]:
    """
    Set the user prompt for the model.
    """
    global_profile = user_data["char_profile"]
    local_story = user_data["local_scene"]
    speech_path = user_data["wav_path"]
    available_traits = user_data["char_style"]

    user_prompt = (
        f"Global Profile: {global_profile}, "
        f"Local Scene: {local_story}, "
        f"Available Traits: {available_traits}"
    )
    return user_prompt, speech_path


def get_archetype_prompt(archetype_data: dict) -> tuple[str, str]:
    """
    Set the archetype prompt for the model.
    """
    archetype_question = archetype_data["question"]
    return archetype_question, archetype_data["wav_path"]

def _handle_role(obj_role):
    """
    This function is used to handle the role.
    The annotator is flexible across all dimensions.
    """
    # Randomly select a dimension
    selected_dimension = random.choice(list(detailed_dimension_prompts.keys()))
    
    # Get the corresponding field name
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    
    # Get the scores for this dimension
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    
    # Calculate the rounded average (excluding None values)
    valid_scores = [score for score in scores if score is not None]
    # if valid_scores:
    #     average_score = round(sum(valid_scores) / len(valid_scores))
    # else:
    #     average_score = 1  # Default score if no valid scores
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases
    if final_score == "N/A":
        final_score = 1
    
    # Get the dimension prompt
    dimension_prompt = detailed_dimension_prompts[selected_dimension]

    user_prompt, _ = get_user_prompt(obj_role)

    final_prompt = f"{dimension_prompt}\n\n{user_prompt}"
    
    # Create the question template
    question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    obj_role["solution"] = f"<answer>{final_score}</answer>"
    return obj_role

def _handle_role_all_fixed_order(obj_role):
    """
    This function is used to handle the role all fixed order.
    The annotator is consistent across all dimensions.
    """
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{user_prompt}\n\n{general_prompt_detailed}"
    final_prompt = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": final_prompt}]}]
    solution_str = ""
    for dimension in FIXED_ORDER:
        field_name = DIMENSION_TO_FIELD_MAPPING_SCHEMA[dimension]
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        try:
            scores = obj_role[value_field_name]
        except KeyError:
            scores = obj_role["annotations"][value_field_name]
        valid_scores = [score for score in scores if score is not None]
        if len(valid_scores) > 0:
            final_score = random.choice(valid_scores)
        else:
            final_score = 1 # all null usually are insufficient / bad cases
        if final_score == "N/A":
            final_score = 1
        solution_str += f"[{field_name}]{final_score}"

    obj_role["solution"] = f"<answer>{solution_str}</answer>"
    
    return obj_role


def _handle_role_all_fixed_order_consistent_annotator(obj_role):
    """
    This function is used to handle the role all fixed order consistent annotator.
    The annotator is consistent across all dimensions.
    """
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{user_prompt}\n\n{general_prompt_detailed}"
    final_prompt = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": final_prompt}]}]
    
    # Determine a consistent position to use across all dimensions
    # First, find the maximum number of valid scores across all dimensions
    max_valid_scores = 0
    for dimension in FIXED_ORDER:
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        try:
            scores = obj_role[value_field_name]
        except KeyError:
            scores = obj_role["annotations"][value_field_name]
        valid_scores = [score for score in scores if score is not None]
        max_valid_scores = max(max_valid_scores, len(valid_scores))
    
    # Choose a consistent position (0-based index)
    if max_valid_scores > 0:
        consistent_position = random.randint(0, max_valid_scores - 1)
    else:
        consistent_position = 0
    
    solution_str = ""
    for dimension in FIXED_ORDER:
        field_name = DIMENSION_TO_FIELD_MAPPING_SCHEMA[dimension]
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        scores = obj_role[value_field_name]
        final_score = scores[consistent_position]
        if final_score is None:
            final_score = 1 # all null usually are insufficient / bad cases
        if final_score == "N/A":
            final_score = 1
        solution_str += f"[{field_name}]{final_score}"

    obj_role["solution"] = f"<answer>{solution_str}</answer>"
    
    return obj_role


def _handle_role_all_flexible_order(obj_role):
    """
    This function is used to handle the role all flexible order.
    The annotator is flexible across all dimensions.
    """
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{user_prompt}\n\n{general_prompt}"
    final_prompt = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": final_prompt}]}]
    random_order = random.sample(FIXED_ORDER, len(FIXED_ORDER))
    solution_str = ""
    for dimension in random_order:
        field_name = DIMENSION_TO_FIELD_MAPPING_SCHEMA[dimension]
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        try:
            scores = obj_role[value_field_name]
        except KeyError:
            scores = obj_role["annotations"][value_field_name]
        valid_scores = [score for score in scores if score is not None]
        if len(valid_scores) > 0:
            final_score = random.choice(valid_scores)
        else:
            final_score = 1 # all null usually are insufficient / bad cases
        if final_score == "N/A":
            final_score = 1
        solution_str += f"[{field_name}]{final_score}"
    obj_role["solution"] = f"<answer>{solution_str}</answer>"
    return obj_role

def _handle_role_all_flexible_order_consistent_annotator(obj_role):
    """
    This function is used to handle the role all flexible order consistent annotator.
    The annotator is consistent across all dimensions.
    """
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{user_prompt}\n\n{general_prompt}"
    final_prompt = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": final_prompt}]}]
    
    # Determine a consistent position to use across all dimensions
    # First, find the maximum number of valid scores across all dimensions
    max_valid_scores = 0
    for dimension in FIXED_ORDER:
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        try:
            scores = obj_role[value_field_name]
        except KeyError:
            scores = obj_role["annotations"][value_field_name]
        valid_scores = [score for score in scores if score is not None]
        max_valid_scores = max(max_valid_scores, len(valid_scores))
    
    # Choose a consistent position (0-based index)
    if max_valid_scores > 0:
        consistent_position = random.randint(0, max_valid_scores - 1)
    else:
        consistent_position = 0
    
    random_order = random.sample(FIXED_ORDER, len(FIXED_ORDER))
    solution_str = ""
    for dimension in random_order:
        field_name = DIMENSION_TO_FIELD_MAPPING_SCHEMA[dimension]
        value_field_name = DIMENSION_TO_FIELD_MAPPING[dimension]
        try:
            scores = obj_role[value_field_name]
        except KeyError:
            scores = obj_role["annotations"][value_field_name]
        final_score = scores[consistent_position]
        if final_score == "N/A":
            final_score = 1
        if final_score is None:
            final_score = 1 # all null usually are insufficient / bad cases
        solution_str += f"[{field_name}]{final_score}"
    obj_role["solution"] = f"<answer>{solution_str}</answer>"
    return obj_role

def _handle_role_caption(obj_role):
    """
    This function is used to handle the role caption.
    """
    # Randomly select a dimension
    selected_dimension = random.choice(list(detailed_dimension_prompts.keys()))
    
    # Get the corresponding field name
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    
    # Get the scores for this dimension
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    
    # Calculate the rounded average (excluding None values)
    valid_scores = [score for score in scores if score is not None]
    # if valid_scores:
    #     average_score = round(sum(valid_scores) / len(valid_scores))
    # else:
    #     average_score = 1  # Default score if no valid scores
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases

    if final_score == "N/A":
        final_score = 1

    # Get the dimension prompt
    dimension_prompt = detailed_dimension_prompts[selected_dimension]

    user_prompt, _ = get_user_prompt(obj_role)

    final_prompt = f"{dimension_prompt}\n\n{user_prompt}\n\n{caption_prompt}"
    
    # Create the question template
    question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    try:
        obj_role["solution"] = f"<think>{obj_role['audio_info']['audio_caption']}</think><answer>{final_score}</answer>"
    except:
        # raise ValueError(f"audio_info is not found in {obj_role}")
        obj_role["solution"] = f"<think>N/A</think><answer>{final_score}</answer>"
    return obj_role


def _handle_role_emotion(obj_role):
    """
    This function is used to handle the role caption.
    """
    # Randomly select a dimension
    selected_dimension = random.choice(list(detailed_dimension_prompts.keys()))
    
    # Get the corresponding field name
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    
    # Get the scores for this dimension
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    
    # Calculate the rounded average (excluding None values)
    valid_scores = [score for score in scores if score is not None]
    # if valid_scores:
    #     average_score = round(sum(valid_scores) / len(valid_scores))
    # else:
    #     average_score = 1  # Default score if no valid scores
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases

    if final_score == "N/A":
        final_score = 1

    # Get the dimension prompt
    dimension_prompt = detailed_dimension_prompts[selected_dimension]

    user_prompt, _ = get_user_prompt(obj_role)

    final_prompt = f"{dimension_prompt}\n\n{user_prompt}\n\n{emotion_prompt}"
    
    # Create the question template
    question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    try:
        obj_role["solution"] = f"<think>{obj_role['audio_info']['emotion_caption']}</think><answer>{final_score}</answer>"
    except:
        # raise ValueError(f"audio_info is not found in {obj_role}")
        obj_role["solution"] = f"<think>N/A</think><answer>{final_score}</answer>"
    return obj_role

def _handle_role_reasoning(obj_role):
    """
    This function is used to handle the role reasoning.
    """
    # Randomly select a variation
    prompt_variation = random.choice(prompt_variations["variations"]["dimension_prompts"])
    selected_dimension = prompt_variation["dimension_name"]
    
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    valid_scores = [score for score in scores if score is not None]
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases

    if final_score == "N/A":
        final_score = 1
        
    dimension_prompt = prompt_variation["prompt_text"]
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{dimension_prompt}\n\n{user_prompt}"
    question_template = f"{final_prompt}\n\n"
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    obj_role["solution"] = f"<answer>{final_score}</answer>"

    return obj_role

def _handle_role_variation(obj_role):
    """
    This function is used to handle the role variation.
    """
    # Randomly select a variation
    prompt_variation = random.choice(prompt_variations["variations"]["dimension_prompts"])
    selected_dimension = prompt_variation["dimension_name"]
    
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    valid_scores = [score for score in scores if score is not None]
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases

    if final_score == "N/A":
        final_score = 1

    dimension_prompt = prompt_variation["prompt_text"]
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{dimension_prompt}\n\n{user_prompt}"
    question_template = f"{final_prompt}\n\n"
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    obj_role["solution"] = f"<answer>{final_score}</answer>"

    return obj_role

def _handle_role_variation_caption(obj_role):
    """
    This function is used to handle the role variation caption.
    """
    # Randomly select a variation
    prompt_variation = random.choice(prompt_variations["variations"]["dimension_prompts"])
    selected_dimension = prompt_variation["dimension_name"]
    
    field_name = DIMENSION_TO_FIELD_MAPPING[selected_dimension]
    try:
        scores = obj_role[field_name]
    except KeyError:
        scores = obj_role["annotations"][field_name]
    valid_scores = [score for score in scores if score is not None]
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases

    if final_score == "N/A":
        final_score = 1
    
    dimension_prompt = prompt_variation["prompt_text"]
    user_prompt, _ = get_user_prompt(obj_role)
    final_prompt = f"{dimension_prompt}\n\n{user_prompt}\n\n{caption_prompt}"
    question_template = f"{final_prompt}\n\n"
    obj_role["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_role["wav_path"]}, {"type": "text", "text": question_template}]}]
    try:
        obj_role["solution"] = f"<think>{obj_role['audio_info']['audio_caption']}</think><answer>{final_score}</answer>"
    except:
        # raise ValueError(f"audio_info is not found in {obj_role}")
        obj_role["solution"] = f"<think>N/A</think><answer>{final_score}</answer>"
    return obj_role


def _handle_archetype(obj_archetype):
    """
    This function is used to handle the archetype.
    """
    # Randomly select a dimension
    selected_dimension = random.choice(list(detailed_archetype_dimension_prompts.keys()))
    
    # Get the scores for this dimension
    try:
        scores = obj_archetype[selected_dimension]
    except KeyError:
        scores = obj_archetype["annotations"][selected_dimension]
    
    # Calculate the rounded average (excluding None values)
    valid_scores = [score if score is not None else 1 for score in scores]
    if len(valid_scores) > 0:
        final_score = random.choice(valid_scores)
    else:
        final_score = 1 # all null usually are insufficient / bad cases
    if isinstance(final_score, bool):
        final_score = int(final_score) # convert bool to int
    
    # Get the dimension prompt
    dimension_prompt = detailed_archetype_dimension_prompts[selected_dimension]

    user_prompt, _ = get_archetype_prompt(obj_archetype)

    final_prompt = f"{dimension_prompt}\n\n{user_prompt}"
    
    # Create the question template
    question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
    
    obj_archetype["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": obj_archetype["wav_path"]}, {"type": "text", "text": question_template}]}]
    obj_archetype["solution"] = f"<answer>{final_score}</answer>"
    return obj_archetype


def handle_json_line(json_line, sample_rate=16000):
    obj = json.loads(json_line)
    waveform = _handle_wav(obj["wav_path"], sample_rate)
    obj["audio"] = waveform.numpy()

    dataset_name = obj.get("dataset_name", "ROLE")

    if dataset_name == "AVQA":
        return _handle_avqa(obj)
    elif dataset_name == "ROLE":
        return _handle_role(obj)
    elif dataset_name == "RoleAllFixOrder":
        return _handle_role_all_fixed_order(obj)
    elif dataset_name == "RoleAllFixOrderConsistent":
        return _handle_role_all_fixed_order_consistent_annotator(obj)
    elif dataset_name == "RoleAllFlexOrder":
        return _handle_role_all_flexible_order(obj)
    elif dataset_name == "RoleAllFlexOrderConsistent":
        return _handle_role_all_flexible_order_consistent_annotator(obj)
    elif dataset_name == "RoleCaption":
        return _handle_role_caption(obj)
    elif dataset_name == "RoleEmotion":
        return _handle_role_emotion(obj)
    elif dataset_name == "RoleReasoning":
        return _handle_role_reasoning(obj)
    elif dataset_name == "RoleVariation":
        return _handle_role_variation(obj)
    elif dataset_name == "RoleVariationCaption":
        return _handle_role_variation_caption(obj)
    elif dataset_name == "Arche":
        return _handle_archetype(obj)
    return obj


class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, is_perturb=False):
        super().__init__()
        self.lists = []
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                self.lists.append(line)

        self.sample_rate = sample_rate
        self.is_perturb = is_perturb
        logging.info(f"{data_file}, len:{len(self.lists)}, rate:{sample_rate}")

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        return handle_json_line(self.lists[index], self.sample_rate)
