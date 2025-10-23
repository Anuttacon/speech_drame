#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate || exit 0
conda activate zeroshot

export PYTHONPATH=/mnt/project/jiatong/project/role_play_baseline/few_shot:${PYTHONPATH}

gemini_model=gemini-2.5-pro

for trial in 1 2 3 4 5 ; do

python3 pyscripts/eval.py \
    --model_name gemini25 \
    --model_tag ${gemini_model} \
    --data_path /mnt/project/jiatong/project/role_play_baseline/base_data/human.jsonl \
    --output_path ${gemini_model}_human_fewshot_detailed_results_trial${trial}.jsonl \
    --prompt_type few_shot_combined \
    --prompt_version v2 \
    --batch_size 10 \
    --num_workers 5 \
    --use_schema true \
    --few_shot_examples_path /mnt/project/jiatong/project/role_play_baseline/base_data/all_train.jsonl \
    --num_few_shot 3 \
    --few_shot_strategy random & 

done
