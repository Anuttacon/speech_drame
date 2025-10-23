import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
from transformers import HfArgumentParser


@dataclass
class AnalysisArguments:
    """
    Arguments for analyzing role evaluation results.
    """
    result_file: Optional[str] = field(default=None, metadata={"help": "path to evaluation result file"})
    output_dir: Optional[str] = field(default="analysis", metadata={"help": "output directory for analysis"})
    evaluation_type: Optional[str] = field(default=None, metadata={"help": "evaluation type to analyze"})

    def __post_init__(self):
        if self.result_file is None:
            raise ValueError("result_file should not be None")


def extract_ratings_from_evaluation(evaluation_text: str) -> Dict[str, float]:
    """Extract numerical ratings from evaluation text."""
    ratings = {}
    
    # Define patterns for different rating categories
    patterns = {
        'voice_identity_matching': r'Voice Identity Matching:\s*(\d+)',
        'trait_embodiment': r'Trait Embodiment:\s*(\d+)',
        'emotional_accuracy': r'Emotional Accuracy:\s*(\d+)',
        'naturalness': r'Naturalness:\s*(\d+)',
        'consistency_with_character_profile': r'Consistency with Character Profile:\s*(\d+)',
        'style_consistency': r'Style Consistency:\s*(\d+)',
        'age_appropriateness': r'Age Appropriateness:\s*(\d+)',
        'personality_alignment': r'Personality Alignment:\s*(\d+)',
        'emotion_recognition': r'Emotion Recognition:\s*(\d+)',
        'emotional_intensity': r'Emotional Intensity:\s*(\d+)',
        'emotional_range': r'Emotional Range:\s*(\d+)',
        'contextual_appropriateness': r'Contextual Appropriateness:\s*(\d+)',
        'pitch_variation': r'Pitch Variation:\s*(\d+)',
        'rhythmic_naturalness': r'Rhythmic Naturalness:\s*(\d+)',
        'dynamic_range': r'Dynamic Range:\s*(\d+)',
        'character_consistency': r'Character Consistency:\s*(\d+)',
        'emotion_intensity_control': r'Emotion Intensity Control:\s*(\d+)',
        'stress_and_emphasis': r'Stress and Emphasis:\s*(\d+)',
        'global_story_coherence': r'Global Story Coherence:\s*(\d+)',
        'local_story_fit': r'Local Story Fit:\s*(\d+)',
        # Add patterns for dimension-specific tags
        'pitch_dynamics': r'<pitch_dynamics>(\d+)</pitch_dynamics>',
        'rhythmic_naturalness': r'<rhythmic_naturalness>(\d+)</rhythmic_naturalness>',
        'stress_emphasis': r'<stress_emphasis>(\d+)</stress_emphasis>',
        'emotion_accuracy': r'<emotion_accuracy>(\d+)</emotion_accuracy>',
        'emotion_intensity': r'<emotion_intensity>(\d+)</emotion_intensity>',
        'emotional_dynamic_range': r'<emotional_dynamic_range>(\d+)</emotional_dynamic_range>',
        'voice_identity_matching': r'<voice_identity_matching>(\d+)</voice_identity_matching>',
        'trait_embodiment': r'<trait_embodiment>(\d+)</trait_embodiment>',
        'local_scene_fit': r'<local_scene_fit>(\d+)</local_scene_fit>',
        'global_story_fit': r'<global_story_fit>(\d+)</global_story_fit>'
    }
    
    for category, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            ratings[category] = float(match.group(1))
    
    return ratings


def calculate_correlation(model_ratings: Dict[str, float], ground_truth: Dict[str, List[int]]) -> Dict[str, float]:
    """Calculate correlation between model ratings and ground truth."""
    correlations = {}
    
    # Map model rating categories to ground truth categories
    category_mapping = {
        'voice_identity_matching': 'voice_identity_matching',
        'trait_embodiment': 'trait_embodiment',
        'emotional_accuracy': 'emotion_accuracy',
        'pitch_variation': 'pitch_variation',
        'rhythmic_naturalness': 'rhythmic_naturalness',
        'dynamic_range': 'dynamic_range',
        'emotion_intensity_control': 'emotion_intensity_control',
        'global_story_coherence': 'global_story_coherence',
        'local_story_fit': 'local_story_fit',
        'stress_and_emphasis': 'stress_and_emphasis',
        # Add mappings for new dimensions
        'pitch_dynamics': 'pitch_variation',
        'stress_emphasis': 'stress_and_emphasis',
        'emotion_intensity': 'emotion_intensity_control',
        'emotional_dynamic_range': 'dynamic_range',
        'local_scene_fit': 'local_story_fit',
        'global_story_fit': 'global_story_coherence'
    }
    
    for model_category, gt_category in category_mapping.items():
        if model_category in model_ratings and gt_category in ground_truth:
            model_rating = model_ratings[model_category]
            gt_ratings = ground_truth[gt_category]
            
            if len(gt_ratings) > 0:
                # Calculate correlation with mean of ground truth ratings
                gt_mean = np.mean(gt_ratings)
                correlation = np.corrcoef([model_rating], [gt_mean])[0, 1]
                correlations[f"{model_category}_vs_{gt_category}"] = correlation
    
    return correlations


def analyze_evaluation_results(results: List[Dict[str, Any]], eval_type: str) -> Dict[str, Any]:
    """Analyze evaluation results and generate statistics."""
    analysis = {
        'evaluation_type': eval_type,
        'total_samples': len(results),
        'successful_evaluations': 0,
        'failed_evaluations': 0,
        'model_ratings_summary': {},
        'ground_truth_summary': {},
        'correlations': [],
        'sample_analysis': [],
        'probability_score_analysis': {}
    }
    
    all_model_ratings = []
    all_ground_truth = {}
    all_probability_scores = []
    
    for result in results:
        evaluation_type = result.get('evaluation_type', '')
        specific_dimension = result.get('specific_dimension', '')
        
        if evaluation_type == "specific_dimension" and specific_dimension:
            # Handle specific dimension evaluation
            model_score = result.get('model_score')
            probability_score = result.get('probability_score')
            evaluation_text = result.get('evaluation_text', '')
            
            if model_score is not None:
                analysis['successful_evaluations'] += 1
                all_model_ratings.append({specific_dimension: model_score})
                
                if probability_score is not None:
                    all_probability_scores.append(probability_score)
                
                # Calculate correlations
                ground_truth_ratings = result.get('ground_truth_ratings', {})
                correlations = calculate_correlation({specific_dimension: model_score}, ground_truth_ratings)
                if correlations:
                    analysis['correlations'].append({
                        'sample_id': result.get('id', ''),
                        'correlations': correlations
                    })
                
                # Sample analysis
                analysis['sample_analysis'].append({
                    'sample_id': result.get('id', ''),
                    'model_score': model_score,
                    'probability_score': probability_score,
                    'evaluation_text': evaluation_text,
                    'ground_truth': ground_truth_ratings,
                    'correlations': correlations
                })
            else:
                analysis['failed_evaluations'] += 1
        else:
            # Handle multi-dimension evaluation
            dimension_scores = result.get('dimension_scores', {})
            ground_truth_ratings = result.get('ground_truth_ratings', {})
            
            if dimension_scores:
                analysis['successful_evaluations'] += 1
                all_model_ratings.append(dimension_scores)
                
                # Calculate correlations
                correlations = calculate_correlation(dimension_scores, ground_truth_ratings)
                if correlations:
                    analysis['correlations'].append({
                        'sample_id': result.get('id', ''),
                        'correlations': correlations
                    })
                
                # Sample analysis
                analysis['sample_analysis'].append({
                    'sample_id': result.get('id', ''),
                    'dimension_scores': dimension_scores,
                    'ground_truth': ground_truth_ratings,
                    'correlations': correlations
                })
            else:
                analysis['failed_evaluations'] += 1
    
    # Calculate summary statistics for model ratings
    if all_model_ratings:
        rating_categories = set()
        for ratings in all_model_ratings:
            rating_categories.update(ratings.keys())
        
        for category in rating_categories:
            values = [ratings.get(category, 0) for ratings in all_model_ratings if category in ratings]
            if values:
                analysis['model_ratings_summary'][category] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
    
    # Calculate summary statistics for ground truth
    for category, ratings_list in all_ground_truth.items():
        if ratings_list:
            analysis['ground_truth_summary'][category] = {
                'mean': np.mean(ratings_list),
                'std': np.std(ratings_list),
                'min': np.min(ratings_list),
                'max': np.max(ratings_list),
                'count': len(ratings_list)
            }
    
    # Analyze probability scores if available
    if all_probability_scores:
        analysis['probability_score_analysis'] = {
            'mean': np.mean(all_probability_scores),
            'std': np.std(all_probability_scores),
            'min': np.min(all_probability_scores),
            'max': np.max(all_probability_scores),
            'count': len(all_probability_scores)
        }
    
    return analysis


def generate_report(analysis: Dict[str, Any], output_file: str):
    """Generate a comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append(f"ROLE EVALUATION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Evaluation Type: {analysis['evaluation_type']}")
    report.append(f"Total Samples: {analysis['total_samples']}")
    report.append(f"Successful Evaluations: {analysis['successful_evaluations']}")
    report.append(f"Failed Evaluations: {analysis['failed_evaluations']}")
    report.append(f"Success Rate: {analysis['successful_evaluations']/analysis['total_samples']*100:.2f}%")
    report.append("")
    
    # Model Ratings Summary
    report.append("MODEL RATINGS SUMMARY:")
    report.append("-" * 40)
    for category, stats in analysis['model_ratings_summary'].items():
        report.append(f"{category}:")
        report.append(f"  Mean: {stats['mean']:.2f}")
        report.append(f"  Std:  {stats['std']:.2f}")
        report.append(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
        report.append(f"  Count: {stats['count']}")
        report.append("")
    
    # Ground Truth Summary
    report.append("GROUND TRUTH SUMMARY:")
    report.append("-" * 40)
    for category, stats in analysis['ground_truth_summary'].items():
        report.append(f"{category}:")
        report.append(f"  Mean: {stats['mean']:.2f}")
        report.append(f"  Std:  {stats['std']:.2f}")
        report.append(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
        report.append(f"  Count: {stats['count']}")
        report.append("")
    
    # Probability Score Analysis
    if analysis['probability_score_analysis']:
        report.append("PROBABILITY SCORE ANALYSIS:")
        report.append("-" * 40)
        prob_stats = analysis['probability_score_analysis']
        report.append(f"Mean: {prob_stats['mean']:.3f}")
        report.append(f"Std:  {prob_stats['std']:.3f}")
        report.append(f"Range: {prob_stats['min']:.3f} - {prob_stats['max']:.3f}")
        report.append(f"Count: {prob_stats['count']}")
        report.append("")
    
    # Correlation Analysis
    if analysis['correlations']:
        report.append("CORRELATION ANALYSIS:")
        report.append("-" * 40)
        
        # Aggregate correlations across all samples
        all_correlations = {}
        for sample_corr in analysis['correlations']:
            for corr_name, corr_value in sample_corr['correlations'].items():
                if corr_name not in all_correlations:
                    all_correlations[corr_name] = []
                all_correlations[corr_name].append(corr_value)
        
        for corr_name, corr_values in all_correlations.items():
            if len(corr_values) > 0:
                mean_corr = np.mean(corr_values)
                std_corr = np.std(corr_values)
                report.append(f"{corr_name}:")
                report.append(f"  Mean Correlation: {mean_corr:.3f}")
                report.append(f"  Std Correlation:  {std_corr:.3f}")
                report.append(f"  Sample Count: {len(corr_values)}")
                report.append("")
    
    # Sample Analysis (first 5 samples)
    report.append("SAMPLE ANALYSIS (First 5 samples):")
    report.append("-" * 40)
    for i, sample in enumerate(analysis['sample_analysis'][:5]):
        report.append(f"Sample {i+1} (ID: {sample['sample_id']}):")
        if 'dimension_scores' in sample:
            report.append(f"  Dimension Scores: {sample['dimension_scores']}")
        if 'model_score' in sample:
            report.append(f"  Model Score: {sample['model_score']}")
        if 'probability_score' in sample:
            report.append(f"  Probability Score: {sample['probability_score']:.3f}")
        if sample['correlations']:
            report.append(f"  Correlations: {sample['correlations']}")
        report.append("")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    # Also print to console
    print('\n'.join(report))


def main():
    parser = HfArgumentParser(AnalysisArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(args)
    
    # Load results
    logging.info(f"Loading results from {args.result_file}")
    with open(args.result_file, 'r') as f:
        results = json.load(f)
    
    # Determine evaluation type
    eval_type = args.evaluation_type
    if not eval_type:
        eval_type = results[0].get('evaluation_type', 'unknown') if results else 'unknown'
    
    # Analyze results
    logging.info(f"Analyzing {len(results)} samples with evaluation type: {eval_type}")
    analysis = analyze_evaluation_results(results, eval_type)
    
    # Generate report
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"analysis_{eval_type}.txt")
    generate_report(analysis, output_file)
    
    # Save analysis as JSON
    analysis_file = os.path.join(args.output_dir, f"analysis_{eval_type}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logging.info(f"Analysis completed. Report saved to {output_file}")
    logging.info(f"Analysis data saved to {analysis_file}")


if __name__ == "__main__":
    main() 