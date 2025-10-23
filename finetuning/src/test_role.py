import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import random
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import transformers
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, HfArgumentParser, Qwen2AudioForConditionalGeneration, GenerationConfig

from dataset.prompt import (
    detailed_dimension_prompts, 
    general_prompt, 
    general_prompt_detailed,
    caption_prompt, 
    emotion_prompt
)
from dataset.dataset import get_user_prompt

# preset for different prompt variations
prompt_variations_path = Path(__file__).parent / "dataset/prompt_variations.json"
with open(prompt_variations_path) as f:
    prompt_variations = json.load(f)
    dimension_prompts = prompt_variations["variations"]["dimension_prompts"]
    dimention2prompt = {}
    for prompt in dimension_prompts:
        if prompt["dimension_name"] not in dimention2prompt:
            dimention2prompt[prompt["dimension_name"]] = [prompt["prompt_text"]]
        else:
            dimention2prompt[prompt["dimension_name"]].append(prompt["prompt_text"])


@dataclass
class RoleTestArguments:
    """
    Arguments for role-related testing.
    """
    model_path: Optional[str] = field(default=None, metadata={"help": "model directory path"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test data file path"})
    out_file: Optional[str] = field(default=None, metadata={"help": "output file path"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "batch size for inference"})
    force: Optional[bool] = field(default=False, metadata={"help": "force regenerate results"})
    evaluation_type: Optional[str] = field(default="specific_dimension", metadata={
        "help": "evaluation type: specific_dimension or comprehensive"
    })
    specific_dimension: Optional[str] = field(default="emotion_accuracy", metadata={
        "help": "specific dimension to evaluate (e.g., pitch_dynamics, emotion_accuracy, etc.)"
    })
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "maximum new tokens for generation"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "temperature for generation"})
    top_k: Optional[int] = field(default=10000, metadata={"help": "top k for generation"})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "top p for generation"})
    use_gt_thinking: Optional[bool] = field(default=False, metadata={"help": "use gt thinking"})

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("model_path should not be None")


def _get_audio(wav_path: str) -> torch.Tensor:
    """Load and preprocess audio file."""
    if not os.path.exists(wav_path):
        # use a small silence tensor
        return torch.zeros(int(0.5 * 16000))
    
    try:
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        return waveform[0]
    except Exception as e:
        logging.error(f"Failed to load audio file {wav_path}: {e}")
        return torch.zeros(int(0.5 * 16000))


def _create_role_evaluation_prompt(data_item: Dict[str, Any], eval_type: str, specific_dimension: Optional[str] = None, use_gt_thinking: Optional[bool] = False) -> tuple[str, str, str]:
    """Create role-specific evaluation prompts using the exact same format as dataset.py."""
    
    if eval_type == "specific_dimension" and specific_dimension:
        if specific_dimension not in detailed_dimension_prompts:
            raise ValueError(f"Unknown dimension: {specific_dimension}")
        
        dataset_name = data_item.get("dataset_name", "ROLE")
        
        if dataset_name == "ROLE" or dataset_name == "RoleVariation":
            # Get the dimension prompt exactly as in dataset.py
            if dataset_name == "RoleVariation":
                dimension_prompt = random.choice(dimention2prompt[specific_dimension])
            else:
                dimension_prompt = detailed_dimension_prompts[specific_dimension]
            
            # Get user prompt exactly as in dataset.py
            user_prompt, speech_path = get_user_prompt(data_item)
            
            # Create final prompt exactly as in dataset.py
            final_prompt = f"{dimension_prompt}\n\n{user_prompt}"
            
            # Create the question template exactly as in dataset.py
            question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
            assistant_content = ""
        elif dataset_name == "RoleCaption" or dataset_name == "RoleVariationCaption":
            if dataset_name == "RoleVariationCaption":
                dimension_prompt = random.choice(dimention2prompt[specific_dimension])
            else:
                dimension_prompt = detailed_dimension_prompts[specific_dimension]
            user_prompt, speech_path = get_user_prompt(data_item)
            final_prompt = f"{dimension_prompt}\n\n{user_prompt}\n\n{caption_prompt}"
            question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
            assistant_content = ""
            if use_gt_thinking:
                # use ground truth thinking information to support the evaluation
                if data_item['audio_info'] is not None:
                    assistant_content = f"<think>{data_item['audio_info']['audio_caption']}</think><"
                else:
                    assistant_content = f"<think>N/A</think><"
        elif dataset_name == "RoleEmotion":
            dimension_prompt = detailed_dimension_prompts[specific_dimension]
            user_prompt, speech_path = get_user_prompt(data_item)
            final_prompt = f"{dimension_prompt}\n\n{user_prompt}\n\n{emotion_prompt}"
            question_template = f"Please evaluate the audio based on the following criteria:\n\n{final_prompt}\n\nOutput the answer in <answer> </answer>."
            assistant_content = ""
            if use_gt_thinking:
                # use ground truth thinking information to support the evaluation
                if data_item['audio_info'] is not None:
                    assistant_content = f"<think>{data_item['audio_info']['audio_caption']}</think><"
                else:
                    assistant_content = f"<think>N/A</think><"
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        return question_template, speech_path, assistant_content
    
    elif eval_type == "comprehensive":

        user_prompt, speech_path = get_user_prompt(data_item)

        # TODO(jiatong): add variable for different dataset name (now fixed)
        general_prompt_base = general_prompt_detailed if data_item["dataset_name"] == "RoleAllFixOrder" else general_prompt
        prompt_template = f"Please evaluate the audio based on the following criteria:\n\n{user_prompt}\n\n{general_prompt_base}\n\nOutput the answer in <answer> </answer>."

        if data_item["dataset_name"] == "RoleCaption":
            prompt_template += f"\n\n{caption_prompt}"
        elif data_item["dataset_name"] == "RoleEmotion":
            prompt_template += f"\n\n{emotion_prompt}"
        
        assistant_content = ""
        return prompt_template, speech_path, assistant_content
    
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")


def _get_message(data_item: Dict[str, Any], eval_type: str, specific_dimension: Optional[str] = None, use_gt_thinking: Optional[bool] = False) -> List[Dict[str, Any]]:
    """Create the message format for the model input using exact same format as dataset.py."""
    prompt, speech_path, assistant_content = _create_role_evaluation_prompt(data_item, eval_type, specific_dimension, use_gt_thinking)
    
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": speech_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Add assistant message with pre-filled content if available
    if assistant_content:
        message.append({
            "role": "assistant", 
            "content": assistant_content
        })
    
    return message


def extract_evaluation(output_str: str, eval_type: str, specific_dimension: Optional[str] = None) -> Dict[str, Any]:
    """Extract evaluation results from model output."""
    
    if eval_type == "specific_dimension" and specific_dimension:
        # Extract single dimension score using <answer></answer> format
        answer_pattern = r"answer>(.*?)</answer>"
        match = re.search(answer_pattern, output_str, re.DOTALL)
        
        if match:
            answer_text = match.group(1).strip()
            # Try to extract numeric score
            score_match = re.search(r'(\d+)', answer_text)
            if score_match:
                score = int(score_match.group(1))
            elif "N/A" in answer_text:
                score = 1
            else:
                score = 1
            return {
                "score": score,
                "evaluation_text": answer_text,
                "raw_output": output_str
            }
        
        return {
            "score": None,
            "evaluation_text": "No valid score found",
            "raw_output": output_str
        }
    
    else:
        # Extract multiple dimension scores
        results = {}
        
        # Extract scores for each dimension
        for dimension in detailed_dimension_prompts.keys():
            pattern = f"{dimension}>(.*?)</{dimension}>"
            match = re.search(pattern, output_str, re.DOTALL)
            if match:
                dimension_text = match.group(1).strip()
                score_match = re.search(r'(\d+)', dimension_text)
                if score_match:
                    results[dimension] = int(score_match.group(1))
                else:
                    results[dimension] = None
            else:
                results[dimension] = None
        
        return {
            "dimension_scores": results,
            "raw_output": output_str
        }


def _get_score_from_token_probabilities(model, inputs, processor, data_args) -> tuple:
    """
    Get the score by analyzing token probabilities for scoring tokens (1-5).
    Dynamically finds <answer> and </answer> tokens and analyzes tokens in between.
    Includes retry mechanism for cases without expected tags.
    
    Args:
        model: The trained model
        inputs: Model inputs
        processor: Audio processor
        data_args: Data arguments
        
    Returns:
        tuple: (batch_probability_scores, batch_responses)
    """
    # Token IDs for "1", "2", "3", "4", "5"
    score_token_ids = []
    for score in ["1", "2", "3", "4", "5"]:
        token_id = processor.tokenizer.encode(score, add_special_tokens=False)
        if token_id:
            score_token_ids.append(token_id[0])
    
    # Get token IDs for "<answer>" and "</answer>"
    answer_start_tokens = processor.tokenizer.encode("answer>", add_special_tokens=False)
    answer_end_tokens = processor.tokenizer.encode("</answer>", add_special_tokens=False)

    force_words_ids = [
        [processor.tokenizer.encode(score, add_special_tokens=False)[0] for score in ["1", "2", "3", "4", "5"]],
        [processor.tokenizer.encode("answer>", add_special_tokens=False)[0]],
        [processor.tokenizer.encode("</answer>", add_special_tokens=False)[0]]
    ]

    # Prepare output list
    batch_probability_scores = []
    batch_responses = []
    
    def try_generate_with_retry(inputs, attempt=0):
        """Helper function to generate with retry mechanism using batch processing"""
        with torch.no_grad():
            # Different strategies for different attempts
            if attempt == 0:
                # First attempt: normal generation
                generation_kwargs = {
                    "max_new_tokens": data_args.max_new_tokens,
                    "temperature": data_args.temperature,
                    "top_k": min(data_args.top_k, 50),  # Cap top_k to avoid generation issues
                    "top_p": data_args.top_p,
                    "do_sample": True,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            elif attempt == 1:
                # Second attempt: enable force_words_ids to encourage proper format
                generation_kwargs = {
                    "max_new_tokens": data_args.max_new_tokens,
                    "temperature": data_args.temperature * 0.8,  # Slightly lower temperature
                    "top_k": min(data_args.top_k, 50),  # Cap top_k to avoid generation issues
                    "top_p": data_args.top_p,
                    "do_sample": True,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            else:
                # Third attempt: greedy search with force_words_ids
                generation_kwargs = {
                    "max_new_tokens": data_args.max_new_tokens,
                    "do_sample": False,  # Greedy search
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "force_words_ids": force_words_ids,
                    "num_beams": 2,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }
            
            outputs = model.generate(**inputs, **generation_kwargs)
            
            # Debug: Check if generation was successful
            if hasattr(outputs, 'sequences') and outputs.sequences is not None:
                print(f"Generation successful for attempt {attempt + 1}, sequences shape: {outputs.sequences.shape}")
            else:
                print(f"Generation failed for attempt {attempt + 1} - no sequences generated")
            
            return outputs
    
    def analyze_generation_output(outputs, idx):
        """Helper function to analyze a single generation output"""
        # Get the generated token sequence first, regardless of scores availability
        generated_ids = outputs.sequences[idx][inputs["input_ids"].size(1):]  # Remove input tokens
        
        # Check if any tokens were generated
        if len(generated_ids) == 0:
            print(f"No tokens generated for item {idx}")
            return None, ""
        
        # print(inputs["input_ids"])
        # print(processor.decode(outputs.sequences[idx], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        # exit(0)
        response = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Debug: Print the generated response
        # print(f"Generated response for item {idx}: '{response[:100]}...' (length: {len(response)})")
        
        # If no scores available, return None score but keep the response
        if not hasattr(outputs, 'scores') or not outputs.scores:
            print(f"No scores available for item {idx}")
            return None, response
        
        # Find the position of <answer> and </answer> tokens
        answer_start_pos = -1
        answer_end_pos = -1
        
        # Search for answer_start_tokens sequence
        for i in range(len(generated_ids) - len(answer_start_tokens) + 1):
            if all(generated_ids[i + j] == answer_start_tokens[j] for j in range(len(answer_start_tokens))):
                answer_start_pos = i + len(answer_start_tokens)
                break
        
        # Search for answer_end_tokens sequence
        for i in range(answer_start_pos, len(generated_ids) - len(answer_end_tokens) + 1):
            if all(generated_ids[i + j] == answer_end_tokens[j] for j in range(len(answer_end_tokens))):
                answer_end_pos = i
                break
        
        if answer_start_pos != -1 and answer_end_pos != -1 and answer_start_pos < answer_end_pos:
            # Get the tokens between <answer> and </answer>
            answer_tokens = generated_ids[answer_start_pos:answer_end_pos]
            
            # Find the first token that corresponds to a score (1-5)
            score_token_pos = -1
            for i, token_id in enumerate(answer_tokens):
                if token_id in score_token_ids:
                    score_token_pos = answer_start_pos + i
                    break
            
            if score_token_pos != -1:
                # Get the logits for the score token position
                sample_scores = [score[idx] for score in outputs.scores if score is not None]
                for step_idx, step_logits in enumerate(sample_scores):
                    if step_idx == score_token_pos:
                        logits = step_logits  # Shape: [vocab_size]
                        
                        # Convert score_token_ids to tensor for indexing
                        score_token_tensor = torch.tensor(score_token_ids, device=logits.device, dtype=torch.long)
                        
                        # Extract logits for scoring tokens (1-5)
                        score_logits = logits[score_token_tensor]  # Shape: [5]

                        # Apply softmax to get probabilities
                        score_probs = F.softmax(score_logits, dim=-1)
                        
                        # Calculate expected score
                        expected_score = torch.dot(score_probs, torch.arange(1, 6, device=logits.device).float())
                        return float(expected_score), response
        
        # If we get here, we didn't find the expected format
        return None, response
    
    # Try multiple generation attempts with batch processing
    success_mask = [False] * len(inputs["input_ids"])
    final_scores = [None] * len(inputs["input_ids"])
    final_responses = [None] * len(inputs["input_ids"])
    
    for attempt in range(3):  # Max 3 attempts
        # Check if we need to continue (some items still not successful)
        if all(success_mask):
            break
            
        print(f"Batch generation attempt {attempt + 1}")
        
        try:
            # Generate for the entire batch
            outputs = try_generate_with_retry(inputs, attempt)
            
            # Analyze each item in the batch
            for idx in range(len(inputs["input_ids"])):
                if success_mask[idx]:  # Skip already successful items
                    continue
                    
                score, response = analyze_generation_output(outputs, idx)
                
                # Check if score is valid (not None and not NaN)
                if score is not None and not (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
                    final_scores[idx] = score
                    final_responses[idx] = response
                    success_mask[idx] = True
                    print(f"Success on attempt {attempt + 1} for item {idx}")
                else:
                    if score is not None:
                        print(f"Attempt {attempt + 1} failed for item {idx} - Invalid score (NaN/Inf): {score}, response: {response[:100] if response else 'None'}...")
                    else:
                        print(f"Attempt {attempt + 1} failed for item {idx}, response: {response[:100] if response else 'None'}...")
                    
                    if "N/A" in response:
                        final_scores[idx] = 1
                        final_responses[idx] = response
                        success_mask[idx] = True
                        print(f"Further check for N/A, success on attempt {attempt + 1} for item {idx}")
                    
        except Exception as e:
            print(f"Error on batch attempt {attempt + 1}: {str(e)}")
            continue
    
    # Handle remaining failed items with fallback
    for idx in range(len(inputs["input_ids"])):
        if not success_mask[idx]:
            print(f"All attempts failed for item {idx}, using fallback method")
            try:
                # Final fallback: generate without special requirements and use last token
                single_inputs = {
                    "input_ids": inputs["input_ids"][idx:idx+1],
                    "attention_mask": inputs["attention_mask"][idx:idx+1] if "attention_mask" in inputs else None,
                }
                if "audio_features" in inputs:
                    single_inputs["audio_features"] = inputs["audio_features"][idx:idx+1]
                single_inputs = {k: v for k, v in single_inputs.items() if v is not None}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **single_inputs,
                        max_new_tokens=data_args.max_new_tokens,
                        do_sample=True,
                        temperature=data_args.temperature,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                if hasattr(outputs, 'scores') and outputs.scores:
                    generated_ids = outputs.sequences[0][single_inputs["input_ids"].size(1):]
                    final_responses[idx] = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    # Use last token logits for scoring
                    logits = outputs.scores[-1][0]  # Shape: [vocab_size]
                    score_token_tensor = torch.tensor(score_token_ids, device=logits.device, dtype=torch.long)
                    score_logits = logits[score_token_tensor]  # Shape: [5]
                    score_probs = F.softmax(score_logits, dim=-1)
                    fallback_score = float(torch.dot(score_probs, torch.arange(1, 6, device=logits.device).float()))
                    
                    # Check if fallback score is valid
                    if not (np.isnan(fallback_score) or np.isinf(fallback_score)):
                        final_scores[idx] = fallback_score
                    else:
                        print(f"Fallback score is invalid (NaN/Inf): {fallback_score}, using default score 3.0")
                        final_scores[idx] = 3.0  # Default neutral score
                else:
                    final_scores[idx] = 3.0  # Default neutral score
                    final_responses[idx] = "No valid response generated"
                    
            except Exception as e:
                print(f"Fallback also failed for item {idx}: {str(e)}")
                final_scores[idx] = 3.0  # Default neutral score
                final_responses[idx] = "Error in generation"
    
    batch_probability_scores = final_scores
    batch_responses = final_responses
    
    print(f"Batch processing complete. Scores: {batch_probability_scores}")
    return batch_probability_scores, batch_responses


def main():
    parser = HfArgumentParser(RoleTestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    # Check if output file already exists
    if not data_args.force and os.path.exists(data_args.out_file) and os.path.getsize(data_args.out_file) > 0:
        logging.info(f"The {data_args.out_file} exists. Do not regenerate it.")
        return
    
    # Create output directory if it doesn't exist
    out_dir = os.path.abspath(os.path.dirname(data_args.out_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load model and processor
    logging.info(f"Loading model from {data_args.model_path}")
    audio_processor = AutoProcessor.from_pretrained(data_args.model_path)
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        data_args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    audio_model.eval()  # Set to evaluation mode

    # Load test data
    logging.info(f"Loading test data from {data_args.data_file}")
    datas = []
    with open(data_args.data_file, "r") as f:
        for line in f:
            datas.append(json.loads(line.strip()))
    
    logging.info(f"Loaded {len(datas)} test samples")
    
    all_outputs = []
    all_probability_scores = []
    batch_size = data_args.batch_size
    
    # Process data in batches
    for i in tqdm(range(0, len(datas), batch_size), desc="Processing batches"):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_audios = []
        
        for bd in batch_data:
            batch_messages.append(_get_message(bd, data_args.evaluation_type, data_args.specific_dimension, data_args.use_gt_thinking))
            audio_path = bd["wav_path"]
            batch_audios.append(_get_audio(audio_path).numpy())
            

        # Apply chat template and process inputs
        text = [
            audio_processor.apply_chat_template(msg, add_generation_prompt=False if data_args.use_gt_thinking else True, tokenize=False)
            for msg in batch_messages
        ]

        # remove the last two added tokens for gt_thinking
        if data_args.use_gt_thinking:
            # specifically remove <|im_end|>\n at the end of the text
            # do not remove other <|im_end|>\n
            for i in range(len(text)):
                if "<|im_end|>\n" == text[i][-11:]:
                    text[i] = text[i][:-11]
        
        inputs = audio_processor(
            text=text, 
            audios=batch_audios, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(audio_model.device)
        
        # Debug: Check input shapes
        # print(f"Input shapes - input_ids: {inputs['input_ids'].shape}, audio_features: {inputs.get('audio_features', 'None')}")
        # print(f"Sample input text: {text[0][:200]}...")


        # Get probability-based scores if evaluating specific dimension
        batch_probability_scores, batch_responses = _get_score_from_token_probabilities(audio_model, inputs, audio_processor, data_args)
        
        all_probability_scores.extend(batch_probability_scores)
        all_outputs.extend(batch_responses)
        
        logging.info(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    # Process results
    final_output = []
    for idx, (input_example, model_output) in enumerate(zip(datas, all_outputs)):
        evaluation_result = extract_evaluation(model_output, data_args.evaluation_type, data_args.specific_dimension)
        
        # Create result dictionary
        result = {
            "id": input_example["id"],
            "original_id": input_example.get("original_id", ""),
            "char_profile": input_example.get("char_profile", ""),
            "char_style": input_example.get("char_style", ""),
            "transcript": input_example.get("transcript", ""),
            "local_scene": input_example.get("local_scene", ""),
            "evaluation_type": data_args.evaluation_type,
            "specific_dimension": data_args.specific_dimension,
            "raw_model_output": evaluation_result["raw_output"]
        }
        
        # Add evaluation results based on type
        if data_args.evaluation_type == "specific_dimension" and data_args.specific_dimension:
            result["model_score"] = evaluation_result["score"]
            result["probability_score"] = all_probability_scores[idx] if idx < len(all_probability_scores) else None
            result["evaluation_text"] = evaluation_result["evaluation_text"]
        else:
            result["dimension_scores"] = evaluation_result["dimension_scores"]
        
        if "pitch_variation" not in input_example:
            input_example = input_example["annotations"] 
        # Add ground truth ratings
        result["ground_truth_ratings"] = {
            "pitch_variation": input_example.get("pitch_variation", []),
            "rhythmic_naturalness": input_example.get("rhythmic_naturalness", []),
            "dynamic_range": input_example.get("dynamic_range", []),
            "emotion_accuracy": input_example.get("emotion_accuracy", []),
            "emotion_intensity_control": input_example.get("emotion_intensity_control", []),
            "global_story_coherence": input_example.get("global_story_coherence", []),
            "local_story_fit": input_example.get("local_story_fit", []),
            "stress_and_emphasis": input_example.get("stress_and_emphasis", []),
            "trait_embodiment": input_example.get("trait_embodiment", []),
            "voice_identity_matching": input_example.get("voice_identity_matching", []),
            "annotator_confidence_rating": input_example.get("annotator_confidence_rating", [])
        }
        
        final_output.append(result)

    # Save results
    output_path = data_args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    logging.info(f"Results saved to {output_path}")
    logging.info(f"Processed {len(final_output)} samples with evaluation type: {data_args.evaluation_type}")


if __name__ == "__main__":
    main() 
