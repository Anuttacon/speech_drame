#!/bin/bash

source /mnt/project/share/.env

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate || exit 0
conda activate zeroshot

for trial in 1 2 3 4 5 ; do

python3 pyscripts/eval.py \
    --model_name gpt4o \
    --model_tag gpt-4o-audio-preview \
    --data_path /mnt/project/jiatong/project/role_play_baseline/base_data/real_test_wav.jsonl \
    --output_path gpt4ofull_real-test_zeroshot_detailed_results_trial${trial}.jsonl \
    --prompt_type combined \
    --prompt_version v2 \
    --batch_size 100 \
    --num_workers 20 \
    --use_schema true  &

done
