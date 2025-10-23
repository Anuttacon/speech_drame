#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate
conda activate role_eval

export PATH=/mnt/project/jiatong/project/conda_env/miniconda3/envs/role_eval/bin:${PATH}

# Configuration
OUT_DIR=exp/sft_lora_all-real-v2
MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
DATA_FILE=data/train_all_real_v2/data.jsonl
YAML_CONFIG=conf/sft_lora.yaml

# GPU and distributed settings
GPU_NUM=2
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32778

# TensorBoard settings
ENABLE_TENSORBOARD=true
TENSORBOARD_LOG_DIR=""  # Leave empty to use default (output_dir/tensorboard_logs)

# PEFT settings (optional - can be configured in YAML)
# PEFT_METHOD="lora"  # Options: lora, qlora, adalora, ia3, prefix_tuning, prompt_tuning
# PEFT_ENABLED="false"  # Set to "true" to enable PEFT

torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train_sft.py \
        --model_name_or_path ${MODEL_NP} \
        --data_file ${DATA_FILE} \
        --out_dir ${OUT_DIR} \
        --config_path conf/ds_zero1.json \
        --use_wandb false \
        --yaml_config ${YAML_CONFIG} \
        --enable_tensorboard ${ENABLE_TENSORBOARD} \
        ${TENSORBOARD_LOG_DIR:+--tensorboard_log_dir ${TENSORBOARD_LOG_DIR}} || exit 1

# Example with PEFT enabled:
# torchrun --nproc_per_node=${GPU_NUM} \
#     --nnodes=${NODE_NUM} \
#     --node-rank=${NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     src/train_sft.py \
#         --model_name_or_path ${MODEL_NP} \
#         --data_file ${DATA_FILE} \
#         --out_dir ${OUT_DIR} \
#         --config_path conf/ds_zero1.json \
#         --use_wandb false \
#         --yaml_config ${YAML_CONFIG} || exit 1
