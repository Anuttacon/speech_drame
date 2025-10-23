#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate
conda activate role_eval
export PATH=/mnt/project/jiatong/project/conda_env/miniconda3/envs/role_eval/bin:${PATH}

# Configuration
MODEL_PATH="exp_arche/sft_lora_addcontent/checkpoint-10000"  # Path to your trained model
DATA_FILE="data/arche_test_all/data_Arche.jsonl"  # Test data file
OUTPUT_DIR="results_arche/sft_lora_addcontent_specific_dimension_checkpoint-10000"

# Evaluation types
# EVAL_TYPES=("specific_dimension" "comprehensive")
EVAL_TYPES=("specific_dimension")

# Specific dimensions for detailed evaluation with exact prompt alignment
DIMENSIONS=("content_pass" "audio_quality" "human_likeness" "appropriateness")

# GPU settings
GPU_NUM=1
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run specific dimension evaluations with exact prompt alignment
echo "Running specific dimension evaluations with exact prompt alignment..."
for dimension in "${DIMENSIONS[@]}"; do
    echo "Running evaluation for dimension: ${dimension}"
    
    OUTPUT_FILE="${OUTPUT_DIR}/arche_eval_${dimension}_exact.json"
    
    torchrun --nproc_per_node=${GPU_NUM} \
        --nnodes=${NODE_NUM} \
        --node-rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/test_arche.py \
            --model_path ${MODEL_PATH} \
            --data_file ${DATA_FILE} \
            --out_file ${OUTPUT_FILE} \
            --evaluation_type specific_dimension \
            --specific_dimension ${dimension} \
            --batch_size 16 \
            --max_new_tokens 256 \
            --use_gt_thinking false \
            --force false || exit 1
    
    echo "Completed evaluation for dimension: ${dimension}"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo "----------------------------------------"
done

# # Run comprehensive evaluation
# echo "Running comprehensive evaluation..."
# OUTPUT_FILE="${OUTPUT_DIR}/role_eval_comprehensive_exact.json"

# torchrun --nproc_per_node=${GPU_NUM} \
#     --nnodes=${NODE_NUM} \
#     --node-rank=${NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     src/test_role.py \
#         --model_path ${MODEL_PATH} \
#         --data_file ${DATA_FILE} \
#         --out_file ${OUTPUT_FILE} \
#         --evaluation_type comprehensive \
#         --batch_size 4 \
#         --max_new_tokens 10 \
#         --force false || exit 1

# echo "Completed comprehensive evaluation"
# echo "Results saved to: ${OUTPUT_FILE}"
# echo "----------------------------------------"

echo "All role evaluations completed!"
echo "Results saved in: ${OUTPUT_DIR}"

# Optional: Run a specific evaluation only
# Uncomment and modify the following lines to run a specific evaluation:

# echo "Running emotion_accuracy evaluation only..."
# torchrun --nproc_per_node=${GPU_NUM} \
#     --nnodes=${NODE_NUM} \
#     --node-rank=${NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     src/test_role.py \
#         --model_path ${MODEL_PATH} \
#         --data_file ${DATA_FILE} \
#         --audio_dir ${AUDIO_DIR} \
#         --out_file ${OUTPUT_DIR}/role_eval_emotion_accuracy_exact.json \
#         --evaluation_type specific_dimension \
#         --specific_dimension emotion_accuracy \
#         --batch_size 4 \
#         --max_new_tokens 10 \
#         --force false || exit 1 
