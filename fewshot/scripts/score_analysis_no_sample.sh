#!/bin/bash

# Score Analysis Script for Multiple Trials
# Usage: ./score_analysis.sh <model_name> <ground_truth_path> [output_dir]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <ground_truth_path> [output_dir]"
    echo "Example: $0 qwen25_omni ../base_data/evalset_v1/annotations.jsonl"
    exit 1
fi

MODEL_NAME=$1
GROUND_TRUTH_PATH=$2
OUTPUT_DIR=${3:-"analysis_results"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the analysis
echo "Running score analysis..."
python pyscripts/score_avg.py \
    --trial_paths "results/${MODEL_NAME}.jsonl" \
    --ground_truth "$GROUND_TRUTH_PATH" \
    --output "$OUTPUT_DIR/${MODEL_NAME}_analysis.json" \
    --visualization_dir "$OUTPUT_DIR/${MODEL_NAME}_plots"

echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}_analysis.json"
echo "Plots saved to: $OUTPUT_DIR/${MODEL_NAME}_plots/" 
