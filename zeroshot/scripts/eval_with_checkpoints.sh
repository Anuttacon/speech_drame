#!/bin/bash

# Example script demonstrating checkpointing functionality for evaluation

# Set up directories
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./results"
mkdir -p $CHECKPOINT_DIR
mkdir -p $OUTPUT_DIR

# Example 1: Run evaluation with checkpointing enabled
echo "=== Example 1: Running evaluation with checkpointing ==="
python zeroshot/pyscripts/eval.py \
    --model_name kimi_audio \
    --model_tag moonshotai/Kimi-Audio-7B \
    --data_path base_data/evalset_v1/adjusted_annotation.jsonl \
    --output_path $OUTPUT_DIR/kimi_audio_results.jsonl \
    --prompt_type separate \
    --prompt_version v1 \
    --batch_size 2 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_interval 5

# Example 2: List available checkpoints
echo -e "\n=== Example 2: Listing available checkpoints ==="
python zeroshot/pyscripts/checkpoint_utils.py list $CHECKPOINT_DIR

# Example 3: Show details of a specific checkpoint (replace with actual filename)
echo -e "\n=== Example 3: Show checkpoint details ==="
# Get the latest checkpoint for kimi_audio
LATEST_CHECKPOINT=$(python zeroshot/pyscripts/checkpoint_utils.py latest $CHECKPOINT_DIR --model_name kimi_audio)
if [ ! -z "$LATEST_CHECKPOINT" ]; then
    python zeroshot/pyscripts/checkpoint_utils.py show "$LATEST_CHECKPOINT"
else
    echo "No checkpoint found for kimi_audio"
fi

# Example 4: Resume from checkpoint
echo -e "\n=== Example 4: Resuming from checkpoint ==="
if [ ! -z "$LATEST_CHECKPOINT" ]; then
    python zeroshot/pyscripts/eval.py \
        --model_name kimi_audio \
        --model_tag moonshotai/Kimi-Audio-7B \
        --data_path base_data/evalset_v1/adjusted_annotation.jsonl \
        --output_path $OUTPUT_DIR/kimi_audio_results_resumed.jsonl \
        --prompt_type separate \
        --prompt_version v1 \
        --batch_size 2 \
        --checkpoint_dir $CHECKPOINT_DIR \
        --checkpoint_interval 5 \
        --resume_from "$LATEST_CHECKPOINT"
else
    echo "No checkpoint available for resuming"
fi

# Example 5: Clean up old checkpoints (dry run)
echo -e "\n=== Example 5: Cleanup old checkpoints (dry run) ==="
python zeroshot/pyscripts/checkpoint_utils.py cleanup $CHECKPOINT_DIR --days 1 --dry_run True

echo -e "\n=== Checkpointing demonstration completed ==="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR" 