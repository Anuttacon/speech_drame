#!/bin/bash

python3 score.py \
  --model_results gemini20flash_evalsetv1_zeroshot_results.jsonl \
  --ground_truth ../base_data/evalset_v1/annotation.jsonl \
  --output gemini20flash_evalsetv1_zeroshot_eval.json

python3 score.py \
  --model_results gemini25pro_evalsetv1_zeroshot_results.jsonl \
  --ground_truth ../base_data/evalset_v1/annotation.jsonl \
  --output gemini25pro_evalsetv1_zeroshot_eval.json

python3 score.py \
  --model_results gemini25flash_evalsetv1_zeroshot_results.jsonl \
  --ground_truth ../base_data/evalset_v1/annotation.jsonl \
  --output gemini25flash_evalsetv1_zeroshot_eval.json

python3  score.py \
   --model_results gpt4o_evalsetv1_zeroshot_detailed_results.jsonl \
   --ground_truth ../base_data/evalset_v1/annotation.jsonl \
   --output gpt4o_evalsetv1_zeroshot_detailed_eval.jsonl


python3 score.py \
  --model_results gemini25pro_evalsetv1_zeroshot_detailed_results.jsonl \
  --ground_truth ../base_data/evalset_v1/annotation.jsonl \
  --output gemini25pro_evalsetv1_zeroshot_detailed_eval.json
