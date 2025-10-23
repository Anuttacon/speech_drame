#!/bin/bash

. /mnt/project/jiatong/project/conda_env/miniconda3/bin/activate || exit 0
conda activate zeroshot

export PYTHONPATH=${PYTHONPATH}:$(pwd)

gemini_model=gemini-2.5-pro

for trial in 4 5 ; do

python3 pyscripts/eval.py \
    --model_name gemini25 \
    --model_tag ${gemini_model} \
    --data_path /mnt/project/jiatong/project/role_play_baseline/base_data/all_test_wav.jsonl \
    --output_path ${gemini_model}_all-test_zeroshot_separate_detailed_results_trial${trial}.jsonl \
    --prompt_type separate \
    --prompt_version v2 \
    --batch_size 1000 \
    --num_workers 200 \
    --checkpoint_dir checkpoint \
    --force_resume true \
    --use_schema true 
    

done
