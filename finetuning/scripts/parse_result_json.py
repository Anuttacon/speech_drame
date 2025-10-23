#!/usr/bin/env python3
"""
Model vs Ground Truth Score Evaluation Script

This script compares model scores and probability scores against 
the average of ground truth ratings for various evaluation dimensions.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

class ScoreEvaluator:
    def __init__(self, json_file_path):
        """Initialize the evaluator with JSON data."""
        self.data = self.load_data(json_file_path)
        self.df = self.prepare_dataframe()
        
    def load_data(self, file_path):
        """Load and clean JSON data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle potentially incomplete JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print("JSON appears incomplete, attempting to parse available data...")
                content = content.strip()
                if not content.endswith(']'):
                    last_comma = content.rfind(',')
                    if last_comma > 0:
                        content = content[:last_comma] + ']'
                data = json.loads(content)
            
            print(f"Successfully loaded {len(data)} records")
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def prepare_dataframe(self):
        """Convert JSON data to DataFrame with calculated ground truth averages."""
        records = []
        
        # Mapping of evaluation dimensions to ground truth dimensions
        dimension_mapping = {
            'pitch_dynamics': 'pitch_variation',
            'rhythmic_naturalness': 'rhythmic_naturalness',
            'emotional_dynamic_range': 'dynamic_range',
            'emotion_accuracy': 'emotion_accuracy',
            'emotion_intensity': 'emotion_intensity_control',
            'global_story_fit': 'global_story_coherence',
            'local_story_fit': 'local_story_fit',
            'stress_emphasis': 'stress_and_emphasis',
            'trait_embodiment': 'trait_embodiment',
            'voice_identity_matching': 'voice_identity_matching'
        }
        
        for record in self.data:
            if not record.get('ground_truth_ratings'):
                continue
                
            # Get the specific dimension being evaluated
            eval_dimension = record.get('specific_dimension')
            ground_truth_dimension = dimension_mapping.get(eval_dimension, eval_dimension)
            
            # Calculate ground truth average for the specific dimension
            gt_ratings = record['ground_truth_ratings'].get(ground_truth_dimension, [])
            valid_ratings = [r for r in gt_ratings if r is not None]
            
            if not valid_ratings:
                continue
                
            gt_average = np.mean(valid_ratings)
            
            # Calculate averages for all dimensions (for comprehensive analysis)
            all_averages = {}
            for dim, ratings in record['ground_truth_ratings'].items():
                if dim != 'annotator_confidence_rating':  # Exclude confidence rating
                    valid_ratings_dim = [r for r in ratings if r is not None]
                    if valid_ratings_dim:
                        all_averages[f'gt_avg_{dim}'] = np.mean(valid_ratings_dim)
            
            record_data = {
                'id': record.get('id'),
                'eval_dimension': eval_dimension,
                'model_score': record.get('model_score'),
                'probability_score': record.get('probability_score'),
                'gt_average': gt_average,
                'gt_count': len(valid_ratings),
                'character': record.get('char_profile', '')[:50] + '...' if record.get('char_profile') else '',
                **all_averages
            }
            
            records.append(record_data)
        
        df = pd.DataFrame(records)
        print(f"Prepared DataFrame with {len(df)} valid records")
        return df
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)
        
        # Pearson correlation
        correlation, p_value = stats.pearsonr(predictions, ground_truth)
        
        # Spearman correlation (rank-based)
        spearman_corr, spearman_p = stats.spearmanr(predictions, ground_truth)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Pearson_r': correlation,
            'Pearson_p': p_value,
            'Spearman_r': spearman_corr,
            'Spearman_p': spearman_p
        }
    
    def evaluate_scores(self):
        """Main evaluation function."""
        if self.df.empty:
            print("No valid data to evaluate")
            return
        
        print("=== SCORE EVALUATION RESULTS ===\n")
        
        # Basic statistics
        print("Dataset Statistics:")
        print(f"Total records: {len(self.df)}")
        print(f"Unique evaluation dimensions: {self.df['eval_dimension'].nunique()}")
        print(f"Dimensions: {list(self.df['eval_dimension'].unique())}")
        print()
        
        # Score distributions
        print("Score Distributions:")
        print("Model Scores:")
        print(self.df['model_score'].describe())
        print("\nProbability Scores:")
        print(self.df['probability_score'].describe())
        print("\nGround Truth Averages:")
        print(self.df['gt_average'].describe())
        print()
        
        # Model Score vs Ground Truth
        print("=== MODEL SCORE vs GROUND TRUTH ===")
        model_metrics = self.calculate_metrics(self.df['model_score'], self.df['gt_average'])
        for metric, value in model_metrics.items():
            print(f"{metric}: {value:.4f}")
        print()
        
        # Probability Score vs Ground Truth
        print("=== PROBABILITY SCORE vs GROUND TRUTH ===")
        prob_metrics = self.calculate_metrics(self.df['probability_score'], self.df['gt_average'])
        for metric, value in prob_metrics.items():
            print(f"{metric}: {value:.4f}")
        print()
        
        # Comparison summary
        print("=== COMPARISON SUMMARY ===")
        print(f"Model Score MAE: {model_metrics['MAE']:.4f}")
        print(f"Probability Score MAE: {prob_metrics['MAE']:.4f}")
        print(f"Better MAE: {'Model Score' if model_metrics['MAE'] < prob_metrics['MAE'] else 'Probability Score'}")
        print()
        print(f"Model Score R²: {model_metrics['R²']:.4f}")
        print(f"Probability Score R²: {prob_metrics['R²']:.4f}")
        print(f"Better R²: {'Model Score' if model_metrics['R²'] > prob_metrics['R²'] else 'Probability Score'}")
        print()
        
        # Analysis by dimension
        if self.df['eval_dimension'].nunique() > 1:
            print("=== ANALYSIS BY DIMENSION ===")
            for dimension in self.df['eval_dimension'].unique():
                dim_data = self.df[self.df['eval_dimension'] == dimension]
                if len(dim_data) > 1:
                    print(f"\n{dimension.upper()} (n={len(dim_data)}):")
                    model_corr = stats.pearsonr(dim_data['model_score'], dim_data['gt_average'])[0]
                    prob_corr = stats.pearsonr(dim_data['probability_score'], dim_data['gt_average'])[0]
                    print(f"  Model Score correlation: {model_corr:.4f}")
                    print(f"  Probability Score correlation: {prob_corr:.4f}")
        
        return model_metrics, prob_metrics
    
    def create_visualizations(self, output_dir="."):
        """Create visualizations for the evaluation and save them."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        
        # First visualization: Scatter plots and residuals
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model vs Ground Truth Score Evaluation', fontsize=16, fontweight='bold')
        
        # Model Score vs Ground Truth
        axes[0, 0].scatter(self.df['gt_average'], self.df['model_score'], alpha=0.6, color='blue')
        axes[0, 0].plot([self.df['gt_average'].min(), self.df['gt_average'].max()], 
                       [self.df['gt_average'].min(), self.df['gt_average'].max()], 
                       'r--', alpha=0.8, label='Perfect Agreement')
        axes[0, 0].set_xlabel('Ground Truth Average')
        axes[0, 0].set_ylabel('Model Score')
        axes[0, 0].set_title('Model Score vs Ground Truth')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Probability Score vs Ground Truth
        axes[0, 1].scatter(self.df['gt_average'], self.df['probability_score'], alpha=0.6, color='green')
        axes[0, 1].plot([self.df['gt_average'].min(), self.df['gt_average'].max()], 
                       [self.df['gt_average'].min(), self.df['gt_average'].max()], 
                       'r--', alpha=0.8, label='Perfect Agreement')
        axes[0, 1].set_xlabel('Ground Truth Average')
        axes[0, 1].set_ylabel('Probability Score')
        axes[0, 1].set_title('Probability Score vs Ground Truth')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals for Model Score
        model_residuals = self.df['model_score'] - self.df['gt_average']
        axes[1, 0].scatter(self.df['gt_average'], model_residuals, alpha=0.6, color='blue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Ground Truth Average')
        axes[1, 0].set_ylabel('Residuals (Model - GT)')
        axes[1, 0].set_title('Model Score Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals for Probability Score
        prob_residuals = self.df['probability_score'] - self.df['gt_average']
        axes[1, 1].scatter(self.df['gt_average'], prob_residuals, alpha=0.6, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Ground Truth Average')
        axes[1, 1].set_ylabel('Residuals (Probability - GT)')
        axes[1, 1].set_title('Probability Score Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_plot_path = os.path.join(output_dir, 'score_evaluation_scatter.png')
        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plots saved to: {scatter_plot_path}")
        
        # Second visualization: Distribution comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['gt_average'], bins=20, alpha=0.7, label='Ground Truth', color='red')
        plt.hist(self.df['model_score'], bins=20, alpha=0.7, label='Model Score', color='blue')
        plt.hist(self.df['probability_score'], bins=20, alpha=0.7, label='Probability Score', color='green')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        box_data = [self.df['gt_average'], self.df['model_score'], self.df['probability_score']]
        plt.boxplot(box_data, labels=['Ground Truth', 'Model Score', 'Probability Score'])
        plt.ylabel('Score')
        plt.title('Score Distributions (Box Plot)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        distribution_plot_path = os.path.join(output_dir, 'score_distributions.png')
        plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plots saved to: {distribution_plot_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model scores against ground truth ratings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing evaluation results"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save visualizations and results"
    )
    
    parser.add_argument(
        "--csv_output",
        type=str,
        default="evaluation_results.csv",
        help="Filename for CSV output of evaluation results"
    )
    
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip creating visualizations"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize evaluator
    evaluator = ScoreEvaluator(args.json_file)
    
    # Run evaluation
    model_metrics, prob_metrics = evaluator.evaluate_scores()
    print("model_metrics: ", model_metrics)
    print("prob_metrics: ", prob_metrics)
    
    # Create visualizations if not disabled
    if not args.no_plots:
        evaluator.create_visualizations(args.output_dir)
    
    # Export results to CSV for further analysis
    csv_path = os.path.join(args.output_dir, args.csv_output)
    evaluator.df.to_csv(csv_path, index=False)
    print(f"Results exported to '{csv_path}'")

if __name__ == "__main__":
    main()