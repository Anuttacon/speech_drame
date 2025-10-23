# Copyright 2025 Jiatong Shi (Anuttacon)

import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
            'voice_identity_matching', 'trait_embodiment', 'local_scene_fit', 'global_story_fit',
        ]
        
        for dim in dimensions:
            if dim in parsed:
                scores[dim] = float(parsed[dim])
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

def load_trial_results(trial_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load results from multiple trials and organize by file ID"""
    all_trials = defaultdict(list)
    
    for trial_path in trial_paths:
        try:
            with open(trial_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    # Extract file ID from audio path
                    audio_path = result.get('audio_path', '')
                    if audio_path != "":
                        file_id = audio_path.split('/')[-1].replace('.wav', '')
                    else:
                        file_id = result.get("id", "")
                    all_trials[file_id].append(result)
        except Exception as e:
            print(f"Error loading trial {trial_path}: {e}")
    
    return dict(all_trials)

def calculate_average_scores(trial_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Calculate average scores across trials for each file and dimension"""
    avg_scores = {}
    dimensions = [
        'pitch_dynamics', 'rhythmic_naturalness', 'stress_emphasis',
        'emotion_accuracy', 'emotion_intensity', 'emotional_dynamic_range',
        'voice_identity_matching', 'trait_embodiment', 'local_scene_fit', 'global_story_fit'
    ]
    
    for file_id, trials in trial_results.items():
        if not trials:
            continue
            
        file_scores = {}
        for dim in dimensions:
            dim_scores = []
            for trial in trials:
                try:
                    response = trial.get('response', None)
                    if response is not None:
                        model_scores = extract_scores_from_response(response)
                    else:
                        model_scores = trial
                    dim_scores.append(model_scores[dim])
                except:
                    continue
    
            if dim_scores:
                file_scores[dim] = np.mean(dim_scores)
                file_scores[f"{dim}_std"] = np.std(dim_scores)
                file_scores[f"{dim}_count"] = len(dim_scores)
            else:
                file_scores[dim] = 1.0  # Default score
                file_scores[f"{dim}_std"] = 0.0
                file_scores[f"{dim}_count"] = 0
        
        avg_scores[file_id] = file_scores
    
    return avg_scores


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
    for i in range(5):
        prefix = f"annotation{i}_"
        try:
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
        except (KeyError, TypeError):
            # print(f"Error extracting ground truth scores for {prefix}")
            continue
    
    # Add final0 scores
    prefix = "final0_"
    try:
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
    except (KeyError, TypeError):
        # print(f"Error extracting ground truth scores for {prefix}")
        pass
    
    return dict(scores)

def calculate_metrics(predictions: List[float], ground_truth: List[float]) -> Dict[str, float]:
    """Calculate comprehensive metrics between predictions and ground truth"""
    if len(predictions) != len(ground_truth) or len(predictions) == 0:
        return {
            'correlation': 0.0, 'spearman': 0.0, 'ktau': 0.0,
            'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')
        }
    
    try:
        # Correlation metrics
        pearson = np.corrcoef(predictions, ground_truth)[0, 1]
        spearman = scipy.stats.spearmanr(predictions, ground_truth).correlation
        ktau = scipy.stats.kendalltau(predictions, ground_truth).correlation
        
        # Error metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth)) ** 2))
        
        # Mean Absolute Percentage Error (avoid division by zero)
        mape = np.mean(np.abs((np.array(predictions) - np.array(ground_truth)) / np.array(ground_truth))) * 100
        
        return {
            'correlation': float(pearson),
            'spearman': float(spearman),
            'ktau': float(ktau),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'correlation': 0.0, 'spearman': 0.0, 'ktau': 0.0,
            'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')
        }

def analyze_results(avg_scores: Dict[str, Dict[str, float]], ground_truth_path: str) -> Dict[str, Any]:
    """Analyze averaged results against ground truth"""
    
    # Load ground truth
    ground_truth = {}
    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ground_truth[data['id'].replace(".wav", "")] = data
    
    dimensions = [
        'pitch_dynamics', 'rhythmic_naturalness', 'stress_emphasis',
        'emotion_accuracy', 'emotion_intensity', 'emotional_dynamic_range',
        'voice_identity_matching', 'trait_embodiment', 'local_scene_fit', 'global_story_fit'
    ]
    
    results = {
        'overall': {},
        'dimensions': {},
        'trial_consistency': {},
        'detailed_analysis': {}
    }
    
    all_predictions = []
    all_ground_truth = []
    
    # Analyze each dimension
    for dim in dimensions:
        predictions = []
        ground_truth_scores = []
        trial_stds = []
        trial_counts = []
        
        for file_id, scores in avg_scores.items():
            if file_id in ground_truth:
                # Extract ground truth (average of all annotators)
                try:
                    gt_scores = extract_ground_truth_scores(ground_truth[file_id])
                    avg_gt_score = np.mean(gt_scores[dim])
                except (KeyError, TypeError):
                    print(f"Error extracting ground truth scores for {file_id}")
                    continue
                if len(gt_scores[dim]) == 0:
                    print(f"No ground truth scores for {file_id}")
                    continue
                ground_truth_scores.append(avg_gt_score)

                predictions.append(scores[dim])
                trial_stds.append(scores.get(f"{dim}_std", 0.0))
                trial_counts.append(scores.get(f"{dim}_count", 0))
                
                all_predictions.append(scores[dim])
                all_ground_truth.append(avg_gt_score)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth_scores)

        if dim == 'pitch_dynamics':
            # print(f"Predictions: {predictions}")
            # print(f"Ground truth: {ground_truth_scores}")
            print(f"Mean prediction: {np.mean(predictions)}")
            print(f"Mean ground truth: {np.mean(ground_truth_scores)}")
            print(f"Std prediction: {np.std(predictions)}")
            print(f"Std ground truth: {np.std(ground_truth_scores)}")
            print(f"Count prediction: {len(predictions)}")
            print(f"Count ground truth: {len(ground_truth_scores)}")
            bad_mask = ~np.isfinite(ground_truth_scores)        # True where arr is nan, +inf or -inf
            idx      = np.where(bad_mask)       # positions of the bad values
            count    = bad_mask.sum()           # how many in total

            print(f"{count} problematic values at indices {idx}")
        
        results['dimensions'][dim] = {
            **metrics,
            'num_samples': len(predictions),
            'predictions': predictions,
            'ground_truth': ground_truth_scores,
            'trial_std_mean': np.mean(trial_stds) if trial_stds else 0.0,
            'trial_std_std': np.std(trial_stds) if trial_stds else 0.0,
            'trial_count_mean': np.mean(trial_counts) if trial_counts else 0.0
        }
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_predictions, all_ground_truth)
    results['overall'] = {
        **overall_metrics,
        'total_samples': len(all_predictions)
    }
    
    # Trial consistency analysis
    results['trial_consistency'] = {
        'avg_trial_count': np.mean([scores.get('pitch_dynamics_count', 0) for scores in avg_scores.values()]),
        'std_trial_count': np.std([scores.get('pitch_dynamics_count', 0) for scores in avg_scores.values()]),
        'avg_std_across_dimensions': np.mean([results['dimensions'][dim]['trial_std_mean'] for dim in dimensions])
    }
    
    return results

def create_visualizations(results: Dict[str, Any], output_dir: str):
    """Create comprehensive visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    dimensions = list(results['dimensions'].keys())
    
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_data = []
    for dim in dimensions:
        correlation_data.append([
            results['dimensions'][dim]['correlation'],
            results['dimensions'][dim]['spearman'],
            results['dimensions'][dim]['ktau']
        ])
    
    correlation_df = pd.DataFrame(
        correlation_data,
        index=dimensions,
        columns=['Pearson', 'Spearman', 'Kendall Tau']
    )
    
    sns.heatmap(correlation_df, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f')
    plt.title('Correlation Metrics by Dimension')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error metrics comparison
    plt.figure(figsize=(15, 5))
    
    mae_values = [results['dimensions'][dim]['mae'] for dim in dimensions]
    rmse_values = [results['dimensions'][dim]['rmse'] for dim in dimensions]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
    plt.xlabel('Dimensions')
    plt.ylabel('Error')
    plt.title('MAE vs RMSE by Dimension')
    plt.xticks(x, dimensions, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    mape_values = [results['dimensions'][dim]['mape'] for dim in dimensions]
    plt.bar(dimensions, mape_values, alpha=0.8, color='orange')
    plt.xlabel('Dimensions')
    plt.ylabel('MAPE (%)')
    plt.title('Mean Absolute Percentage Error by Dimension')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Trial consistency analysis
    plt.figure(figsize=(12, 5))
    
    trial_stds = [results['dimensions'][dim]['trial_std_mean'] for dim in dimensions]
    trial_counts = [results['dimensions'][dim]['trial_count_mean'] for dim in dimensions]
    
    plt.subplot(1, 2, 1)
    plt.bar(dimensions, trial_stds, alpha=0.8, color='green')
    plt.xlabel('Dimensions')
    plt.ylabel('Average Trial Std Dev')
    plt.title('Trial Consistency by Dimension')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(dimensions, trial_counts, alpha=0.8, color='purple')
    plt.xlabel('Dimensions')
    plt.ylabel('Average Trial Count')
    plt.title('Trial Count by Dimension')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'trial_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_results(results: Dict[str, Any]):
    """Print comprehensive analysis results"""
    print("=" * 100)
    print("COMPREHENSIVE SCORE ANALYSIS RESULTS")
    print("=" * 100)
    
    # Overall results
    overall = results['overall']
    print(f"\nOVERALL METRICS:")
    print(f"  Total Samples: {overall['total_samples']}")
    print(f"  Pearson Correlation: {overall['correlation']:.4f}")
    print(f"  Spearman Correlation: {overall['spearman']:.4f}")
    print(f"  Kendall Tau: {overall['ktau']:.4f}")
    print(f"  Mean Absolute Error: {overall['mae']:.4f}")
    print(f"  Root Mean Square Error: {overall['rmse']:.4f}")
    print(f"  Mean Absolute Percentage Error: {overall['mape']:.2f}%")
    
    # Trial consistency
    consistency = results['trial_consistency']
    print(f"\nTRIAL CONSISTENCY:")
    print(f"  Average Trial Count: {consistency['avg_trial_count']:.2f} ± {consistency['std_trial_count']:.2f}")
    print(f"  Average Std Dev Across Dimensions: {consistency['avg_std_across_dimensions']:.4f}")
    
    # Per-dimension results
    print(f"\nPER-DIMENSION METRICS:")
    print(f"{'Dimension':<25} {'Pearson':<10} {'Spearman':<10} {'Ktau':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Trial_Std':<10}")
    print("-" * 110)
    
    for dim, metrics in results['dimensions'].items():
        print(f"{dim:<25} {metrics['correlation']:<10.4f} {metrics['spearman']:<10.4f} {metrics['ktau']:<8.4f} "
              f"{metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} {metrics['mape']:<8.2f} {metrics['trial_std_mean']:<10.4f}")
    
    # Summary statistics
    correlations = [metrics['correlation'] for metrics in results['dimensions'].values()]
    spearmans = [metrics['spearman'] for metrics in results['dimensions'].values()]
    ktaus = [metrics['ktau'] for metrics in results['dimensions'].values()]
    maes = [metrics['mae'] for metrics in results['dimensions'].values()]
    rmses = [metrics['rmse'] for metrics in results['dimensions'].values()]
    mapes = [metrics['mape'] for metrics in results['dimensions'].values()]
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Average Pearson: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"  Average Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
    print(f"  Average Ktau: {np.mean(ktaus):.4f} ± {np.std(ktaus):.4f}")
    print(f"  Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Average RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  Average MAPE: {np.mean(mapes):.2f}% ± {np.std(mapes):.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze averaged results from multiple trials")
    parser.add_argument("--trial_paths", type=str, nargs='+', required=True,
                       help="Paths to trial result JSONL files")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth annotation JSONL file")
    parser.add_argument("--output", type=str, default="analysis_results.json",
                       help="Path to save detailed analysis results JSON file")
    parser.add_argument("--visualization_dir", type=str, default="analysis_plots",
                       help="Directory to save visualization plots")
    
    args = parser.parse_args()
    
    print(f"Loading {len(args.trial_paths)} trial results...")
    
    # Load and combine trial results
    trial_results = load_trial_results(args.trial_paths)
    print(f"Loaded results for {len(trial_results)} files")
   

    # Calculate average scores
    print("Calculating average scores across trials...")
    avg_scores = calculate_average_scores(trial_results)
   
    # Analyze results
    print("Analyzing results against ground truth...")
    results = analyze_results(avg_scores, args.ground_truth)
    
    # Print results
    print_detailed_results(results)
    
    # Create visualizations
    print(f"Creating visualizations in {args.visualization_dir}...")
    create_visualizations(results, args.visualization_dir)
    
    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")
    print(f"Visualizations saved to: {args.visualization_dir}/")

if __name__ == "__main__":
    main()
