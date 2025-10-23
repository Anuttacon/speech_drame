# Copyright 2025 Jiatong Shi (Anuttacon)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.qwen25_omni import Qwen25Omni
from models.gemini25_pro import Gemini25Pro
from models.gpt4o import GPT4oAudio
from models.qwen2_audio import Qwen2Audio
from models.kimi_audio import KimiAudio
from utils import str2bool

from prompts.all_prompt import get_user_prompt, get_system_prompt, get_rubric_prompt
from prompts.all_archetype_prompt import get_archetype_user_prompt, get_archetype_rubric_prompt
from prompts.sep_prompt import get_dimension_prompt, get_all_dimension_keys
from prompts.sep_archetype_prompt import get_archetype_dimension_prompt, get_all_archetype_dimension_keys


from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, default="qwen25")
    parser.add_argument("--model_tag", type=str, required=True, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--data_path", type=str, required=True, default="evalset_v1/adjusted_annotation.jsonl")
    parser.add_argument("--output_path", type=str, required=True, default="qwen25_evalset_v1_zeroshot_results.jsonl")
    parser.add_argument("--prompt_type", type=str, choices=["separate", "combined", "archetype_combined", "archetype_separate"], required=True, default="separate")
    parser.add_argument("--prompt_version", type=str, choices=["v1", "v2"], required=True, default="v1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prompt_sample", type=str2bool, default=False) # NOTE(jiatong): not activated yet
    parser.add_argument("--use_schema", type=str2bool, default=True, help="Use schema validation for better output alignment")
    
    # Checkpointing arguments
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                       help="Directory to save checkpoints. If None, no checkpointing is done.")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save checkpoint every N batches")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--force_resume", type=str2bool, default=False,
                       help="Force resume even if output file already exists")
    
    return parser

def save_checkpoint(checkpoint_path: str, results: list, processed_indices: set, 
                   args: argparse.Namespace, start_time: float):
    """Save checkpoint with current progress"""
    checkpoint_data = {
        "args": vars(args),
        "results": results,
        "processed_indices": list(processed_indices),
        "timestamp": time.time(),
        "start_time": start_time,
        "total_processed": len(processed_indices)
    }
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and return data"""
    with open(checkpoint_path, "r") as f:
        checkpoint_data = json.load(f)
    
    # Convert processed_indices back to set
    checkpoint_data["processed_indices"] = set(checkpoint_data["processed_indices"])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from {len(checkpoint_data['processed_indices'])} processed items")
    
    return checkpoint_data

def get_checkpoint_path(checkpoint_dir: str, model_name: str, prompt_type: str, prompt_version: str) -> str:
    """Generate checkpoint file path"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_{model_name}_{prompt_type}_{prompt_version}_{timestamp}.json"
    return os.path.join(checkpoint_dir, filename)

def supports_content_pass(model_name: str) -> bool:
    """Check if a model supports content_pass mode (0-1 prediction)"""
    return model_name in ["kimi_audio", "qwen2audio", "qwen25"]

def main():
    args = get_parser().parse_args()
    print(args)
    
    # Check if we should resume
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_data = load_checkpoint(args.resume_from)
        
        # Override args with checkpoint args (except for resume-specific args)
        checkpoint_args = argparse.Namespace(**checkpoint_data["args"])
        args.model_name = checkpoint_args.model_name
        args.model_tag = checkpoint_args.model_tag
        args.data_path = checkpoint_args.data_path
        args.prompt_type = checkpoint_args.prompt_type
        args.prompt_version = checkpoint_args.prompt_version
        args.batch_size = checkpoint_args.batch_size
        args.num_workers = checkpoint_args.num_workers
        args.use_schema = checkpoint_args.use_schema
        
        # Load existing results and processed indices
        results = checkpoint_data["results"]
        processed_indices = checkpoint_data["processed_indices"]
        start_time = checkpoint_data["start_time"]
        
        print(f"Resuming with {len(results)} existing results")
    else:
        # Check if output file already exists
        if os.path.exists(args.output_path) and not args.force_resume:
            print(f"Warning: Output file {args.output_path} already exists!")
            print("Use --force_resume True to overwrite or --resume_from <checkpoint> to resume")
            return
        
        results = []
        processed_indices = set()
        start_time = time.time()

    assert args.prompt_sample == False, "Prompt sampling is not implemented yet"
    
    # Print content_pass support info
    if supports_content_pass(args.model_name):
        print(f"Model {args.model_name} supports content_pass mode (0-1 prediction)")
    else:
        print(f"Model {args.model_name} does not support content_pass mode (will use 1-5 scoring for all dimensions)")

    # Load data
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Load model
    if args.model_name == "qwen25":
        model = Qwen25Omni(args.model_name, args.model_tag, get_system_prompt())
    elif args.model_name == "qwen2audio":
        model = Qwen2Audio(args.model_name, args.model_tag, get_system_prompt())
    elif args.model_name == "kimi_audio":
        model = KimiAudio(args.model_name, args.model_tag, get_system_prompt())
    elif args.model_name == "gemini25":
        model = Gemini25Pro(args.model_name, args.model_tag, get_system_prompt())
    elif args.model_name == "gpt4o":
        model = GPT4oAudio(args.model_name, args.model_tag, get_system_prompt())
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    
    # Process data in batches
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(0, total_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(data))
        
        # Skip if this batch was already processed
        if all(i in processed_indices for i in range(start_idx, end_idx)):
            print(f"Skipping batch {batch_idx} (already processed)")
            continue
        
        batch = data[start_idx:end_idx]
        
        if args.prompt_type == "combined":
            # Prepare batch inputs
            prompts, audio_paths = [], []
            for item in batch:
                user_prompt, speech_path = get_user_prompt(item)
                rubric_prompt = get_rubric_prompt(detailed=args.prompt_version == "v2")
                full_prompt = rubric_prompt + "\n" + user_prompt
                prompts.append(full_prompt)
                audio_paths.append(speech_path)
            
            # Generate results for the batch
            if args.use_schema and hasattr(model, 'evaluate_with_schema'):
                # Use schema-based evaluation for models that support it
                batch_results = []
                for prompt, audio_path in zip(prompts, audio_paths):
                    result = model.evaluate_with_schema(audio_path, prompt)
                    batch_results.append(result)
            else:
                # Use regular batch generation
                batch_results = model.batch_generate(prompts, audio_paths, num_workers=args.num_workers)
            results.extend(batch_results)
            
        elif args.prompt_type == "separate":
            # Iterate over all dimensions
            dimension_results = {}
            for key in get_all_dimension_keys():
                print(f"Processing dimension: {key}")
                prompts, audio_paths = [], []
                for item in batch:
                    rubric_prompt = get_dimension_prompt(key, detailed=args.prompt_version == "v2")
                    user_prompt, speech_path = get_user_prompt(item)
                    full_prompt = user_prompt + "\n" + rubric_prompt
                    prompts.append(full_prompt)
                    audio_paths.append(speech_path)
                
                # Generate results for the batch
                if args.use_schema and hasattr(model, 'evaluate_with_schema'):
                    # Use schema-based evaluation for models that support it
                    batch_results = []
                    for prompt, audio_path in zip(prompts, audio_paths):
                        result = model.evaluate_with_schema(audio_path, prompt, key)
                        batch_results.append(result)
                else:
                    # Use regular batch generation
                    batch_results = model.batch_generate(prompts, audio_paths, num_workers=args.num_workers)
                dimension_results[key] = batch_results
                print(f"Dimension {key} results: {batch_results}")
            
            # Combine related dimension scores
            aggregated_results = []
            for i, item in enumerate(batch):
                aggregated_result = {
                    "id": item.get("id", f"batch_{start_idx + i}"),
                    "model_name": args.model_name,
                    "response": {},
                }
                for key in get_all_dimension_keys():
                    # Set audio path from dimensional results
                    if "audio_path" not in aggregated_result:
                        aggregated_result["audio_path"] = dimension_results[key][i].get("audio_path", "")

                    if key in dimension_results and i < len(dimension_results[key]):
                        result = dimension_results[key][i]
                        if not result.get("error"):
                            try:
                                aggregated_result["response"][key] = float(result["response"])
                            except:
                                aggregated_result["response"][key] = 1  # Default score on parsing error
                        else:
                            aggregated_result["response"][key] = 1  # Default score on error
                    else:
                        aggregated_result["response"][key] = 1  # Default score if missing
                    
                aggregated_result["response"] = json.dumps(aggregated_result["response"])
                aggregated_results.append(aggregated_result)
            results.extend(aggregated_results)
        
        elif args.prompt_type == "archetype_combined":
            # Prepare batch inputs
            prompts, audio_paths = [], []
            for item in batch:
                user_prompt, speech_path = get_archetype_user_prompt(item)
                rubric_prompt = get_archetype_rubric_prompt(detailed=args.prompt_version == "v2")
                full_prompt = rubric_prompt + "\n" + user_prompt
                prompts.append(full_prompt)
                audio_paths.append(speech_path)

            # Generate results for the batch
            if args.use_schema and hasattr(model, 'evaluate_with_schema'):
                # Use schema-based evaluation for models that support it
                batch_results = []
                for prompt, audio_path in zip(prompts, audio_paths):
                    result = model.evaluate_with_schema(audio_path, prompt, key="archetype")
                    batch_results.append(result)
            else:
                # Use regular batch generation
                batch_results = model.batch_generate(prompts, audio_paths, num_workers=args.num_workers)
            print(f"Archetype results: {batch_results}",flush=True)
            results.extend(batch_results)

        elif args.prompt_type == "archetype_separate":
            # Iterate over all dimensions
            dimension_results = {}
            for key in get_all_archetype_dimension_keys():
                # Automatically detect content_pass dimension and check if model supports it
                is_content_pass = (key == "content_pass" and supports_content_pass(args.model_name))
                if is_content_pass:
                    print(f"Processing dimension: {key} (content_pass mode: 0-1 prediction)")
                else:
                    print(f"Processing dimension: {key}")
                prompts, audio_paths = [], []
                for item in batch:
                    rubric_prompt = get_archetype_dimension_prompt(key, detailed=args.prompt_version == "v2")
                    user_prompt, speech_path = get_archetype_user_prompt(item)
                    full_prompt = user_prompt + "\n" + rubric_prompt
                    prompts.append(full_prompt)
                    audio_paths.append(speech_path)
                
                # Generate results for the batch
                if args.use_schema and hasattr(model, 'evaluate_with_schema'):
                    # Use schema-based evaluation for models that support it
                    batch_results = []
                    for prompt, audio_path in zip(prompts, audio_paths):
                        if is_content_pass:
                            result = model.evaluate_with_schema(audio_path, prompt, is_content_pass=True)
                        else:
                            result = model.evaluate_with_schema(audio_path, prompt, key)
                        batch_results.append(result)
                else:
                    # Use regular batch generation
                    if is_content_pass:
                        batch_results = model.batch_generate(prompts, audio_paths, num_workers=args.num_workers, is_content_pass=True)
                    else:
                        batch_results = model.batch_generate(prompts, audio_paths, num_workers=args.num_workers)
                dimension_results[key] = batch_results
                print(f"Dimension {key} results: {batch_results}")
            
            # Combine related dimension scores
            aggregated_results = []
            for i, item in enumerate(batch):
                aggregated_result = {
                    "id": item.get("id", f"batch_{start_idx + i}"),
                    "model_name": args.model_name,
                    "response": {},
                }
                for key in get_all_archetype_dimension_keys():
                    # Set audio path from dimensional results
                    if "audio_path" not in aggregated_result:
                        aggregated_result["audio_path"] = dimension_results[key][i].get("audio_path", "")

                    if key in dimension_results and i < len(dimension_results[key]):
                        result = dimension_results[key][i]
                        if not result.get("error"):
                            try:
                                aggregated_result["response"][key] = float(result["response"])
                            except:
                                # Default values based on dimension type
                                if key == "content_pass" and supports_content_pass(args.model_name):
                                    aggregated_result["response"][key] = 1.0  # Default to pass for content_pass
                                else:
                                    aggregated_result["response"][key] = 1  # Default score for other dimensions
                        else:
                            # Default values based on dimension type
                            if key == "content_pass" and supports_content_pass(args.model_name):
                                aggregated_result["response"][key] = 1.0  # Default to pass for content_pass
                            else:
                                aggregated_result["response"][key] = 1  # Default score for other dimensions
                    else:
                        # Default values based on dimension type
                        if key == "content_pass" and supports_content_pass(args.model_name):
                            aggregated_result["response"][key] = 1.0  # Default to pass for content_pass
                        else:
                            aggregated_result["response"][key] = 1  # Default score for other dimensions
                    
                aggregated_result["response"] = json.dumps(aggregated_result["response"])
                aggregated_results.append(aggregated_result)
            results.extend(aggregated_results)

        else:
            raise ValueError(f"Prompt type {args.prompt_type} not supported")
        
        # Mark this batch as processed
        for i in range(start_idx, end_idx):
            processed_indices.add(i)
        
        # Save checkpoint if enabled
        if args.checkpoint_dir and (batch_idx + 1) % args.checkpoint_interval == 0:
            checkpoint_path = get_checkpoint_path(
                args.checkpoint_dir, args.model_name, args.prompt_type, args.prompt_version
            )
            save_checkpoint(checkpoint_path, results, processed_indices, args, start_time)
        
        # Save intermediate results to output file
        with open(args.output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    
    # Final save
    with open(args.output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Save final checkpoint
    if args.checkpoint_dir:
        checkpoint_path = get_checkpoint_path(
            args.checkpoint_dir, args.model_name, args.prompt_type, args.prompt_version
        )
        save_checkpoint(checkpoint_path, results, processed_indices, args, start_time)
    
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(results)} items")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()