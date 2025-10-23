#!/usr/bin/env python3
"""
Multi-Dimension Prediction Results Analysis Script

This script processes multiple JSON files from a folder, where each file contains
prediction results for a single dimension, and provides comprehensive statistics
and analysis of the prediction results across all dimensions.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import os
import glob
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

class MultiDimensionPredictionAnalyzer:
    def __init__(self, folder_path):
        """Initialize the analyzer with a folder containing JSON files."""
        self.folder_path = Path(folder_path)
        self.json_files = self.find_json_files()
        self.all_data = []
        self.df = pd.DataFrame()
        
    def find_json_files(self):
        """Find all JSON files in the specified folder."""
        json_files = list(self.folder_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {self.folder_path}")
            return []
        
        print(f"Found {len(json_files)} JSON files:")
        for file in json_files:
            print(f"  - {file.name}")
        return json_files
    
    def load_data_from_file(self, file_path):
        """Load and clean JSON data from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle potentially incomplete JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"JSON appears incomplete in {file_path.name}, attempting to parse available data...")
                content = content.strip()
                if not content.endswith(']'):
                    last_comma = content.rfind(',')
                    if last_comma > 0:
                        content = content[:last_comma] + ']'
                data = json.loads(content)
            
            print(f"Successfully loaded {len(data)} records from {file_path.name}")
            return data
            
        except Exception as e:
            print(f"Error loading data from {file_path.name}: {e}")
            return []
    
    def load_all_data(self):
        """Load data from all JSON files."""
        self.all_data = []
        
        for json_file in self.json_files:
            file_data = self.load_data_from_file(json_file)
            if file_data:
                # Add source file information to each record
                for record in file_data:
                    record['source_file'] = json_file.name
                self.all_data.extend(file_data)
        
        print(f"Total records loaded: {len(self.all_data)}")
        return self.all_data
    
    
    def prepare_dataframe(self):
        """Convert JSON data to DataFrame with prediction results only."""
        if not self.all_data:
            self.load_all_data()
        
        records = []
        
        for record in self.all_data:
            # Get the specific dimension being evaluated
            eval_dimension = record.get('specific_dimension')
            
            # Only include records that have prediction scores
            if record.get('model_score') is None and record.get('probability_score') is None:
                continue
            
            record_data = {
                'id': record.get('id'),
                'source_file': record.get('source_file'),
                'eval_dimension': eval_dimension,
                'model_score': record.get('model_score'),
                'probability_score': record.get('probability_score'),
                'character': record.get('char_profile', '')[:50] + '...' if record.get('char_profile') else '',
                'text': record.get('text', '')[:100] + '...' if record.get('text') else ''
            }
            
            records.append(record_data)
        
        self.df = pd.DataFrame(records)
        print(f"Prepared DataFrame with {len(self.df)} valid prediction records")
        return self.df
    

    def calculate_prediction_statistics(self, predictions):
        """Calculate descriptive statistics for prediction scores."""
        if len(predictions) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'range': np.nan
            }

        try:
            # Convert to numpy array and remove NaN values
            pred_array = np.array(predictions)
            pred_array = pred_array[~np.isnan(pred_array)]
            
            if len(pred_array) == 0:
                return {
                    'count': 0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'range': np.nan
                }
            
            return {
                'count': len(pred_array),
                'mean': np.mean(pred_array),
                'std': np.std(pred_array),
                'min': np.min(pred_array),
                'max': np.max(pred_array),
                'median': np.median(pred_array),
                'q25': np.percentile(pred_array, 25),
                'q75': np.percentile(pred_array, 75),
                'range': np.max(pred_array) - np.min(pred_array)
            }
        except Exception as e:
            print(f"Error calculating prediction statistics: {e}")
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'median': 0,
                'q25': 0,
                'q75': 0,
                'range': 0
            }
    
    def analyze_predictions(self):
        """Main analysis function focusing on prediction statistics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        if self.df.empty:
            print("No valid prediction data to analyze")
            return {}, {}
        
        print("=== MULTI-DIMENSION PREDICTION ANALYSIS RESULTS ===\n")
        
        # Basic statistics
        print("Dataset Statistics:")
        print(f"Total records: {len(self.df)}")
        print(f"Source files: {self.df['source_file'].nunique()}")
        print(f"Files: {list(self.df['source_file'].unique())}")
        print(f"Unique evaluation dimensions: {self.df['eval_dimension'].nunique()}")
        print(f"Dimensions: {list(self.df['eval_dimension'].unique())}")
        print()
        
        # Records per dimension
        print("Records per dimension:")
        dimension_counts = self.df['eval_dimension'].value_counts()
        for dim, count in dimension_counts.items():
            print(f"  {dim}: {count}")
        print()
        
        # Overall prediction statistics
        print("=== OVERALL PREDICTION STATISTICS (ALL DIMENSIONS) ===")
        
        # Model score statistics
        model_scores = self.df['model_score'].dropna()
        if len(model_scores) > 0:
            model_stats = self.calculate_prediction_statistics(model_scores)
            print("Model Score Statistics:")
            print(f"  Count: {model_stats['count']}")
            print(f"  Mean: {model_stats['mean']:.4f}")
            print(f"  Std: {model_stats['std']:.4f}")
            print(f"  Min: {model_stats['min']:.4f}")
            print(f"  Max: {model_stats['max']:.4f}")
            print(f"  Median: {model_stats['median']:.4f}")
            print(f"  Q25: {model_stats['q25']:.4f}")
            print(f"  Q75: {model_stats['q75']:.4f}")
            print(f"  Range: {model_stats['range']:.4f}")
        else:
            print("No model scores available")
            model_stats = {}
        
        print()
        
        # Probability score statistics
        prob_scores = self.df['probability_score'].dropna()
        if len(prob_scores) > 0:
            prob_stats = self.calculate_prediction_statistics(prob_scores)
            print("Probability Score Statistics:")
            print(f"  Count: {prob_stats['count']}")
            print(f"  Mean: {prob_stats['mean']:.4f}")
            print(f"  Std: {prob_stats['std']:.4f}")
            print(f"  Min: {prob_stats['min']:.4f}")
            print(f"  Max: {prob_stats['max']:.4f}")
            print(f"  Median: {prob_stats['median']:.4f}")
            print(f"  Q25: {prob_stats['q25']:.4f}")
            print(f"  Q75: {prob_stats['q75']:.4f}")
            print(f"  Range: {prob_stats['range']:.4f}")
        else:
            print("No probability scores available")
            prob_stats = {}
        
        print()
        
        # Dimension-specific analysis
        print("=== DIMENSION-SPECIFIC PREDICTION ANALYSIS ===")
        dimension_results = {}
        
        for dimension in self.df['eval_dimension'].unique():
            dim_data = self.df[self.df['eval_dimension'] == dimension]
            
            if len(dim_data) > 0:
                print(f"\n{dimension.upper()} (n={len(dim_data)}):")
                
                # Model score statistics for this dimension
                dim_model_scores = dim_data['model_score'].dropna()
                if len(dim_model_scores) > 0:
                    dim_model_stats = self.calculate_prediction_statistics(dim_model_scores)
                    print(f"  Model Score - Mean: {dim_model_stats['mean']:.4f}, Std: {dim_model_stats['std']:.4f}, Range: {dim_model_stats['range']:.4f}")
                else:
                    print("  No model scores for this dimension")
                    dim_model_stats = {}
                
                # Probability score statistics for this dimension
                dim_prob_scores = dim_data['probability_score'].dropna()
                if len(dim_prob_scores) > 0:
                    dim_prob_stats = self.calculate_prediction_statistics(dim_prob_scores)
                    print(f"  Probability Score - Mean: {dim_prob_stats['mean']:.4f}, Std: {dim_prob_stats['std']:.4f}, Range: {dim_prob_stats['range']:.4f}")
                else:
                    print("  No probability scores for this dimension")
                    dim_prob_stats = {}
                
                dimension_results[dimension] = {
                    'model_stats': dim_model_stats,
                    'prob_stats': dim_prob_stats,
                    'count': len(dim_data)
                }
        
        return model_stats, prob_stats, dimension_results
    
    def create_visualizations(self, output_dir="."):
        """Create comprehensive visualizations focusing on prediction distributions and statistics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        if self.df.empty:
            print("No data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        
        # 1. Overall prediction distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Prediction Score Distributions', fontsize=16, fontweight='bold')
        
        # Model Score distribution
        model_scores = self.df['model_score'].dropna()
        if len(model_scores) > 0:
            axes[0, 0].hist(model_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(model_scores.mean(), color='red', linestyle='--', label=f'Mean: {model_scores.mean():.3f}')
            axes[0, 0].axvline(model_scores.median(), color='orange', linestyle='--', label=f'Median: {model_scores.median():.3f}')
            axes[0, 0].set_xlabel('Model Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Model Score Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Probability Score distribution
        prob_scores = self.df['probability_score'].dropna()
        if len(prob_scores) > 0:
            axes[0, 1].hist(prob_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(prob_scores.mean(), color='red', linestyle='--', label=f'Mean: {prob_scores.mean():.3f}')
            axes[0, 1].axvline(prob_scores.median(), color='orange', linestyle='--', label=f'Median: {prob_scores.median():.3f}')
            axes[0, 1].set_xlabel('Probability Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Probability Score Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Box plots for model scores by dimension
        dimensions = self.df['eval_dimension'].unique()
        model_data_by_dim = []
        dim_labels = []
        for dim in dimensions:
            dim_data = self.df[self.df['eval_dimension'] == dim]['model_score'].dropna()
            if len(dim_data) > 0:
                model_data_by_dim.append(dim_data)
                dim_labels.append(dim)
        
        if model_data_by_dim:
            axes[1, 0].boxplot(model_data_by_dim, labels=dim_labels)
            axes[1, 0].set_ylabel('Model Score')
            axes[1, 0].set_title('Model Score Distribution by Dimension')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Box plots for probability scores by dimension
        prob_data_by_dim = []
        dim_labels = []
        for dim in dimensions:
            dim_data = self.df[self.df['eval_dimension'] == dim]['probability_score'].dropna()
            if len(dim_data) > 0:
                prob_data_by_dim.append(dim_data)
                dim_labels.append(dim)
        
        if prob_data_by_dim:
            axes[1, 1].boxplot(prob_data_by_dim, labels=dim_labels)
            axes[1, 1].set_ylabel('Probability Score')
            axes[1, 1].set_title('Probability Score Distribution by Dimension')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        distribution_plot_path = os.path.join(output_dir, 'prediction_distributions.png')
        plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Prediction distribution plots saved to: {distribution_plot_path}")
        
        # 2. Statistics comparison across dimensions
        if len(dimensions) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Prepare statistics data
            model_means = []
            prob_means = []
            model_stds = []
            prob_stds = []
            model_medians = []
            prob_medians = []
            model_ranges = []
            prob_ranges = []
            
            for dimension in dimensions:
                dim_data = self.df[self.df['eval_dimension'] == dimension]
                
                # Model score statistics
                model_scores = dim_data['model_score'].dropna()
                if len(model_scores) > 0:
                    model_means.append(model_scores.mean())
                    model_stds.append(model_scores.std())
                    model_medians.append(model_scores.median())
                    model_ranges.append(model_scores.max() - model_scores.min())
                else:
                    model_means.append(0)
                    model_stds.append(0)
                    model_medians.append(0)
                    model_ranges.append(0)
                
                # Probability score statistics
                prob_scores = dim_data['probability_score'].dropna()
                if len(prob_scores) > 0:
                    prob_means.append(prob_scores.mean())
                    prob_stds.append(prob_scores.std())
                    prob_medians.append(prob_scores.median())
                    prob_ranges.append(prob_scores.max() - prob_scores.min())
                else:
                    prob_means.append(0)
                    prob_stds.append(0)
                    prob_medians.append(0)
                    prob_ranges.append(0)
            
            # Mean comparison
            x = np.arange(len(dimensions))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, model_means, width, label='Model Score', alpha=0.8, color='blue')
            axes[0, 0].bar(x + width/2, prob_means, width, label='Probability Score', alpha=0.8, color='green')
            axes[0, 0].set_ylabel('Mean Score')
            axes[0, 0].set_title('Mean Scores by Dimension')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(dimensions, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Standard deviation comparison
            axes[0, 1].bar(x - width/2, model_stds, width, label='Model Score', alpha=0.8, color='blue')
            axes[0, 1].bar(x + width/2, prob_stds, width, label='Probability Score', alpha=0.8, color='green')
            axes[0, 1].set_ylabel('Standard Deviation')
            axes[0, 1].set_title('Standard Deviation by Dimension')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(dimensions, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Median comparison
            axes[1, 0].bar(x - width/2, model_medians, width, label='Model Score', alpha=0.8, color='blue')
            axes[1, 0].bar(x + width/2, prob_medians, width, label='Probability Score', alpha=0.8, color='green')
            axes[1, 0].set_ylabel('Median Score')
            axes[1, 0].set_title('Median Scores by Dimension')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(dimensions, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Range comparison
            axes[1, 1].bar(x - width/2, model_ranges, width, label='Model Score', alpha=0.8, color='blue')
            axes[1, 1].bar(x + width/2, prob_ranges, width, label='Probability Score', alpha=0.8, color='green')
            axes[1, 1].set_ylabel('Score Range')
            axes[1, 1].set_title('Score Range by Dimension')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(dimensions, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            statistics_comparison_path = os.path.join(output_dir, 'statistics_comparison.png')
            plt.savefig(statistics_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Statistics comparison plots saved to: {statistics_comparison_path}")
        
        # 3. Scatter plot of model vs probability scores
        if len(model_scores) > 0 and len(prob_scores) > 0:
            plt.figure(figsize=(10, 8))
            plt.scatter(model_scores, prob_scores, alpha=0.6, color='purple')
            plt.xlabel('Model Score')
            plt.ylabel('Probability Score')
            plt.title('Model Score vs Probability Score')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(model_scores) > 1 and len(prob_scores) > 1:
                correlation = np.corrcoef(model_scores, prob_scores)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            scatter_plot_path = os.path.join(output_dir, 'model_vs_probability_scatter.png')
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Model vs Probability scatter plot saved to: {scatter_plot_path}")

    def export_detailed_results(self, output_dir=".", filename="detailed_results.csv"):
        """Export detailed results including dimension-specific prediction statistics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        # Export main dataframe
        main_csv_path = os.path.join(output_dir, filename)
        self.df.to_csv(main_csv_path, index=False)
        print(f"Main results exported to: {main_csv_path}")
        
        # Create summary statistics with focus on prediction statistics
        summary_data = []
        
        # Overall statistics
        model_scores = self.df['model_score'].dropna()
        prob_scores = self.df['probability_score'].dropna()
        
        overall_model_stats = self.calculate_prediction_statistics(model_scores) if len(model_scores) > 0 else {}
        overall_prob_stats = self.calculate_prediction_statistics(prob_scores) if len(prob_scores) > 0 else {}
        
        summary_data.append({
            'dimension': 'OVERALL',
            'count': len(self.df),
            'model_count': overall_model_stats.get('count', 0),
            'model_mean': overall_model_stats.get('mean', np.nan),
            'model_std': overall_model_stats.get('std', np.nan),
            'model_min': overall_model_stats.get('min', np.nan),
            'model_max': overall_model_stats.get('max', np.nan),
            'model_median': overall_model_stats.get('median', np.nan),
            'model_range': overall_model_stats.get('range', np.nan),
            'prob_count': overall_prob_stats.get('count', 0),
            'prob_mean': overall_prob_stats.get('mean', np.nan),
            'prob_std': overall_prob_stats.get('std', np.nan),
            'prob_min': overall_prob_stats.get('min', np.nan),
            'prob_max': overall_prob_stats.get('max', np.nan),
            'prob_median': overall_prob_stats.get('median', np.nan),
            'prob_range': overall_prob_stats.get('range', np.nan)
        })
        
        # Dimension-specific statistics
        for dimension in self.df['eval_dimension'].unique():
            dim_data = self.df[self.df['eval_dimension'] == dimension]
            
            if len(dim_data) > 0:
                dim_model_scores = dim_data['model_score'].dropna()
                dim_prob_scores = dim_data['probability_score'].dropna()
                
                dim_model_stats = self.calculate_prediction_statistics(dim_model_scores) if len(dim_model_scores) > 0 else {}
                dim_prob_stats = self.calculate_prediction_statistics(dim_prob_scores) if len(dim_prob_scores) > 0 else {}
                
                summary_data.append({
                    'dimension': dimension,
                    'count': len(dim_data),
                    'model_count': dim_model_stats.get('count', 0),
                    'model_mean': dim_model_stats.get('mean', np.nan),
                    'model_std': dim_model_stats.get('std', np.nan),
                    'model_min': dim_model_stats.get('min', np.nan),
                    'model_max': dim_model_stats.get('max', np.nan),
                    'model_median': dim_model_stats.get('median', np.nan),
                    'model_range': dim_model_stats.get('range', np.nan),
                    'prob_count': dim_prob_stats.get('count', 0),
                    'prob_mean': dim_prob_stats.get('mean', np.nan),
                    'prob_std': dim_prob_stats.get('std', np.nan),
                    'prob_min': dim_prob_stats.get('min', np.nan),
                    'prob_max': dim_prob_stats.get('max', np.nan),
                    'prob_median': dim_prob_stats.get('median', np.nan),
                    'prob_range': dim_prob_stats.get('range', np.nan)
                })
        
        # Export summary
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'prediction_statistics_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Prediction statistics summary exported to: {summary_csv_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze prediction results across multiple dimensions and generate statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing JSON files with prediction results"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations and results"
    )
    
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip creating visualizations"
    )
    
    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Create detailed CSV outputs with dimension-specific prediction statistics"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = MultiDimensionPredictionAnalyzer(args.folder)
    
    if not analyzer.json_files:
        print("No JSON files found to process.")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = args.folder
    
    # Run analysis
    try:
        overall_model_stats, overall_prob_stats, dimension_results = analyzer.analyze_predictions()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualizations if not disabled
        if not args.no_plots:
            analyzer.create_visualizations(args.output_dir)
        
        # Export detailed results if requested
        if args.detailed_output:
            analyzer.export_detailed_results(args.output_dir)
        else:
            # Export basic results
            basic_csv_path = os.path.join(args.output_dir, 'prediction_results.csv')
            analyzer.df.to_csv(basic_csv_path, index=False)
            print(f"Basic prediction results exported to: {basic_csv_path}")
        
        print(f"\nPrediction analysis complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
