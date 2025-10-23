#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate || exit 0
conda activate zeroshot

export PYTHONPATH=/mnt/project/jiatong/project/role_play_baseline/zeroshot:${PYTHONPATH}

gemini_model=gemini-2.5-pro

for trial in 1 2 3 4 5 ; do

python3 pyscripts/eval.py \
    --model_name gemini25 \
    --model_tag ${gemini_model} \
    --data_path /mnt/project/jiatong/project/role_play_baseline/arche_data/test.jsonl \
    --output_path ${gemini_model}_arche-test_zeroshot_detailed_results_trial${trial}.jsonl \
    --prompt_type archetype_combined \
    --prompt_version v2 \
    --batch_size 100 \
    --num_workers 20 \
    --use_schema true & 

done
