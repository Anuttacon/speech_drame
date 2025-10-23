#!/bin/bash

# Score Analysis Script for Multiple Trials
# Usage: ./score_analysis.sh <model_name> <ground_truth_path> [output_dir]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <ground_truth_path> [output_dir]"
    echo "Example: $0 qwen25_omni ../arche_data/test.jsonl"
    exit 1
fi

MODEL_NAME=$1
GROUND_TRUTH_PATH=$2
OUTPUT_DIR=${3:-"analysis_results"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all trial result files for the model
TRIAL_FILES=()
for i in {1..5}; do
    TRIAL_FILE="results/${MODEL_NAME}_trial${i}.jsonl"
    if [ -f "$TRIAL_FILE" ]; then
        TRIAL_FILES+=("$TRIAL_FILE")
        echo "Found trial file: $TRIAL_FILE"
    else
        echo "Warning: Trial file $TRIAL_FILE not found"
    fi
done

if [ ${#TRIAL_FILES[@]} -eq 0 ]; then
    echo "Error: No trial files found for model $MODEL_NAME"
    exit 1
fi

echo "Found ${#TRIAL_FILES[@]} trial files"

# Run the analysis
echo "Running score analysis..."
python pyscripts/arche_score_avg.py \
    --trial_paths "${TRIAL_FILES[@]}" \
    --ground_truth "$GROUND_TRUTH_PATH" \
    --output "$OUTPUT_DIR/${MODEL_NAME}_analysis.json" \
    --visualization_dir "$OUTPUT_DIR/${MODEL_NAME}_plots"

echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}_analysis.json"
echo "Plots saved to: $OUTPUT_DIR/${MODEL_NAME}_plots/" 
