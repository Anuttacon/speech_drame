# Copyright 2025 Jiatong Shi (Anuttacon)

import argparse
import json
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import re
import scipy.stats

def extract_scores_from_response(response: str) -> Dict[str, int]:
    """Extract scores from model response"""
    try:
        # Try to parse as JSON first
        if isinstance(response, str):
            # Remove any markdown formatting
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            parsed = json.loads(response)
        else:
            parsed = response
            
        # Extract the 10 evaluation dimensions
        scores = {}
        dimensions = [
            'pitch_dynamics', 'rhythmic_naturalness', 'stress_emphasis',
            'emotion_accuracy', 'emotion_intensity', 'emotional_dynamic_range',
            'voice_identity_matching', 'trait_embodiment', 'local_scene_fit', 'global_story_fit'
        ]
        
        for dim in dimensions:
            if dim in parsed:
                scores[dim] = int(parsed[dim])
            else:
                scores[dim] = 1  # Default score if missing
                
        return scores
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response[:200]}...")
        # Return default scores on parsing error
        return {
            'pitch_dynamics': 1, 'rhythmic_naturalness': 1, 'stress_emphasis': 1,
            'emotion_accuracy': 1, 'emotion_intensity': 1, 'emotional_dynamic_range': 1,
            'voice_identity_matching': 1, 'trait_embodiment': 1, 'local_scene_fit': 1, 'global_story_fit': 1
        }

def extract_ground_truth_scores(annotation_data: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract ground truth scores from annotation data"""
    scores = defaultdict(list)
    def safe_int_convert(value):
        """Safely convert value to int, handling 'N/A' and other edge cases"""
        if isinstance(value, list):
            value = value[0] if value else '1'
        if isinstance(value, str):
            if value.upper() == 'N/A' or value == '':
                return 1  # Default score for missing data
            try:
                return int(value)
            except ValueError:
                return 1  # Default score for invalid data
        elif value is None:
            return 1
        return int(value)
    
    def safe_list_convert(list_value):
        """Safely convert list value to int, handling 'N/A' and other edge cases"""
        if isinstance(list_value, list):
            return [safe_int_convert(item) for item in list_value]
        return safe_int_convert(list_value)

    source_dimensions = ["pitch_dynamics", "rhythmic_naturalness", "stress_emphasis", "emotion_accuracy", "emotion_intensity", "emotional_dynamic_range", "voice_identity_matching", "trait_embodiment", "local_scene_fit", "global_story_fit", "semantic_matchness"]

    mapping_dimensions = {
        "pitch_dynamics": "pitch_variation",
        "rhythmic_naturalness": "rhythmic_naturalness",
        "stress_emphasis": "stress_and_emphasis",
        "emotion_accuracy": "emotion_accuracy",
        "emotion_intensity": "emotion_intensity_control",
        "emotional_dynamic_range": "dynamic_range",
        "voice_identity_matching": "voice_identity_matching",
        "trait_embodiment": "trait_embodiment",
        "local_scene_fit": "local_story_fit",
        "global_story_fit": "global_story_coherence",
        "semantic_matchness": "semantic_match"
    }


    for dim in source_dimensions:
        try:
            scores[dim] = safe_list_convert(annotation_data[mapping_dimensions[dim]])
        except (KeyError):
            scores[dim] = safe_list_convert(annotation_data["annotations"][mapping_dimensions[dim]])
    return scores



def extract_ground_truth_scores_raw(annotation_data: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract ground truth scores from annotation data"""
    scores = defaultdict(list)
    
    def safe_int_convert(value):
        """Safely convert value to int, handling 'N/A' and other edge cases"""
        if isinstance(value, list):
            value = value[0] if value else '1'
        if isinstance(value, str):
            if value.upper() == 'N/A' or value == '':
                return 1  # Default score for missing data
            try:
                return int(value)
            except ValueError:
                return 1  # Default score for invalid data
        return int(value)
    
    # Extract scores from all annotators (annotation0-4 and final0)
    for i in range(3):
        prefix = f"annotation{i}_"
        scores['pitch_dynamics'].append(safe_int_convert(annotation_data[f"{prefix}pitch_variation"]))
        scores['rhythmic_naturalness'].append(safe_int_convert(annotation_data[f"{prefix}rhythmic_naturalness"]))
        scores['stress_emphasis'].append(safe_int_convert(annotation_data[f"{prefix}stress_and_emphasis"]))
        scores['emotion_accuracy'].append(safe_int_convert(annotation_data[f"{prefix}emotion_accuracy"]))
        scores['emotion_intensity'].append(safe_int_convert(annotation_data[f"{prefix}emotion_intensity_control"]))
        scores['emotional_dynamic_range'].append(safe_int_convert(annotation_data[f"{prefix}dynamic_range"]))
        scores['voice_identity_matching'].append(safe_int_convert(annotation_data[f"{prefix}voice_identity_matching"]))
        scores['trait_embodiment'].append(safe_int_convert(annotation_data[f"{prefix}trait_embodiment"]))
        scores['local_scene_fit'].append(safe_int_convert(annotation_data[f"{prefix}local_story_fit"]))
        scores['global_story_fit'].append(safe_int_convert(annotation_data[f"{prefix}global_story_coherence"]))
    
    return dict(scores)

def calculate_correlation(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculate Pearson correlation between predictions and ground truth"""
    if len(predictions) != len(ground_truth):
        return 0.0
    
    try:
        return np.corrcoef(predictions, ground_truth)[0, 1]
    except:
        return 0.0

def calculate_spearman(predictions: list, ground_truth: list) -> float:
    if len(predictions) != len(ground_truth):
        return 0.0
    try:
        return float(scipy.stats.spearmanr(predictions, ground_truth).correlation)
    except Exception:
        return 0.0

def calculate_ktau(predictions: list, ground_truth: list) -> float:
    if len(predictions) != len(ground_truth):
        return 0.0
    try:
        return float(scipy.stats.kendalltau(predictions, ground_truth).correlation)
    except Exception:
        return 0.0

def calculate_mae(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculate Mean Absolute Error"""
    if len(predictions) != len(ground_truth):
        return float('inf')
    
    return float(np.mean(np.abs(np.array(predictions) - np.array(ground_truth))))

def calculate_rmse(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculate Root Mean Square Error"""
    if len(predictions) != len(ground_truth):
        return float('inf')
    
    return float(np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth)) ** 2)))

def evaluate_results(model_results_path: str, ground_truth_path: str) -> Dict[str, Any]:
    """Evaluate model results against ground truth"""
    
    # Load model results
    model_results = []
    with open(model_results_path, 'r') as f:
        for line in f:
            model_results.append(json.loads(line.strip()))
    
    # Load ground truth
    ground_truth = {}
    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ground_truth[data['id'].replace(".wav", "")] = data
    
    # Extract file IDs from model results
    file_ids = []
    for result in model_results:
        # Extract file ID from audio path
        audio_path = result.get('audio_path', '')
        if audio_path != "":
            file_id = audio_path.split('/')[-1].replace('.wav', '')
        else:
            file_id = result.get("id", "")
        file_ids.append(file_id)
    
    # Calculate metrics for each dimension
    dimensions = [
        'pitch_dynamics', 'rhythmic_naturalness', 'stress_emphasis',
        'emotion_accuracy', 'emotion_intensity', 'emotional_dynamic_range',
        'voice_identity_matching', 'trait_embodiment', 'local_scene_fit', 'global_story_fit'
    ]
    
    results = {
        'overall': {
            'correlation': 0.0,
            'spearman': 0.0,
            'ktau': 0.0,
            'mae': 0.0,
            'rmse': 0.0,
            'total_samples': 0
        },
        'dimensions': {}
    }
    
    all_predictions = []
    all_ground_truth = []
    
    for dim in dimensions:
        predictions = []
        ground_truth_scores = []
        
        for i, result in enumerate(model_results):
            file_id = file_ids[i]
            
            if file_id in ground_truth:
                # Extract model prediction
                response = result.get('response', None)
                if response is not None:
                    model_scores = extract_scores_from_response(response)
                else:
                    model_scores = result
                predictions.append(model_scores[dim])
                
                # Extract ground truth (average of all annotators)
                gt_scores = extract_ground_truth_scores(ground_truth[file_id])
                avg_gt_score = np.mean(gt_scores[dim])
                ground_truth_scores.append(avg_gt_score)
                
                all_predictions.append(model_scores[dim])
                all_ground_truth.append(avg_gt_score)
        
        # Calculate metrics for this dimension
        correlation = calculate_correlation(predictions, ground_truth_scores)
        spearman = calculate_spearman(predictions, ground_truth_scores)
        ktau = calculate_ktau(predictions, ground_truth_scores)
        mae = calculate_mae(predictions, ground_truth_scores)
        rmse = calculate_rmse(predictions, ground_truth_scores)
        
        results['dimensions'][dim] = {
            'correlation': correlation,
            'spearman': spearman,
            'ktau': ktau,
            'mae': mae,
            'rmse': rmse,
            'num_samples': len(predictions),
            'predictions': predictions,
            'ground_truth': ground_truth_scores
        }
    
    # Calculate overall metrics
    results['overall']['correlation'] = calculate_correlation(all_predictions, all_ground_truth)
    results['overall']['spearman'] = calculate_spearman(all_predictions, all_ground_truth)
    results['overall']['ktau'] = calculate_ktau(all_predictions, all_ground_truth)
    results['overall']['mae'] = calculate_mae(all_predictions, all_ground_truth)
    results['overall']['rmse'] = calculate_rmse(all_predictions, all_ground_truth)
    results['overall']['total_samples'] = len(all_predictions)
    
    return results

def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way"""
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall results
    overall = results['overall']
    print(f"\nOVERALL METRICS:")
    print(f"  Total Samples: {overall['total_samples']}")
    print(f"  Pearson Correlation: {overall['correlation']:.4f}")
    print(f"  Spearman Correlation: {overall['spearman']:.4f}")
    print(f"  Kendall Tau: {overall['ktau']:.4f}")
    print(f"  Mean Absolute Error: {overall['mae']:.4f}")
    print(f"  Root Mean Square Error: {overall['rmse']:.4f}")
    
    # Per-dimension results
    print(f"\nPER-DIMENSION METRICS:")
    print(f"{'Dimension':<25} {'Pearson':<10} {'Spearman':<10} {'Ktau':<8} {'MAE':<8} {'RMSE':<8} {'Samples':<8}")
    print("-" * 90)
    
    for dim, metrics in results['dimensions'].items():
        print(f"{dim:<25} {metrics['correlation']:<10.4f} {metrics['spearman']:<10.4f} {metrics['ktau']:<8.4f} {metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} {metrics['num_samples']:<8}")
    
    # Summary statistics
    correlations = [metrics['correlation'] for metrics in results['dimensions'].values()]
    spearmans = [metrics['spearman'] for metrics in results['dimensions'].values()]
    ktaus = [metrics['ktau'] for metrics in results['dimensions'].values()]
    maes = [metrics['mae'] for metrics in results['dimensions'].values()]
    rmses = [metrics['rmse'] for metrics in results['dimensions'].values()]
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Average Pearson: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"  Average Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
    print(f"  Average Ktau: {np.mean(ktaus):.4f} ± {np.std(ktaus):.4f}")
    print(f"  Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Average RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model results against ground truth")
    parser.add_argument("--model_results", type=str, required=True, 
                       help="Path to model results JSONL file")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth annotation JSONL file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save detailed results JSON file")
    
    args = parser.parse_args()
    
    # Evaluate results
    results = evaluate_results(args.model_results, args.ground_truth)
    
    # Print results
    print_results(results)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
