#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate || exit 0
conda activate zeroshot

export PYTHONPATH=${PYTHONPATH}:/mnt/project/jiatong/project/role_play_baseline/zeroshot


python3 pyscripts/eval.py \
    --model_name qwen2audio \
    --model_tag  Qwen/Qwen2-Audio-7B \
    --data_path /mnt/project/jiatong/project/role_play_baseline/base_data/real_test_wav.jsonl \
    --output_path qwen2audio-7b_real-test_zeroshot_detailedresults.jsonl \
    --prompt_type separate \
    --prompt_version v2 \
    --batch_size 1 \
    --num_workers 1 \
    --use_schema false

