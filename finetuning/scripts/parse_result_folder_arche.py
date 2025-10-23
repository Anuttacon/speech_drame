#!/usr/bin/env python3
"""
Multi-Dimension Model vs Ground Truth Score Evaluation Script

This script processes multiple JSON files from a folder, where each file contains
evaluations for a single dimension, and provides comprehensive analysis across
all dimensions with focus on correlation metrics.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import argparse
import os
import glob
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

class MultiDimensionScoreEvaluator:
    def __init__(self, folder_path):
        """Initialize the evaluator with a folder containing JSON files."""
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
    
    def parse_ratings(self, ratings, dimension=None):
        """Parse ratings from a list of ratings."""
        valid_ratings = []
        for r in ratings:
            if r is None:
                continue
            if r == "N/A":
                if dimension == "content_pass":
                    valid_ratings.append(False)  # Treat N/A as False for content_pass
                else:
                    valid_ratings.append(1)
            else:
                valid_ratings.append(r)
        return valid_ratings
    
    def compute_majority_vote(self, ratings):
        """Compute majority vote for boolean ratings (content_pass)."""
        if not ratings:
            return None
        
        true_count = sum(1 for r in ratings if r is True)
        false_count = sum(1 for r in ratings if r is False)
        
        if true_count > false_count:
            return True
        elif false_count > true_count:
            return False
        else:
            # Tie case - return None or could return a default
            return None
    
    def calculate_classification_metrics(self, predictions, ground_truth, threshold=0.5):
        """Calculate classification metrics for binary classification (content_pass)."""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan,
                'auc': np.nan,
                'confusion_matrix': None
            }
        
        try:
            # Convert to lists first to handle pandas Series and mixed types
            if hasattr(predictions, 'tolist'):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)
            
            if hasattr(ground_truth, 'tolist'):
                ground_truth_list = ground_truth.tolist()
            else:
                ground_truth_list = list(ground_truth)
            
            # Convert to numpy arrays with explicit dtype handling
            predictions_array = np.array(predictions_list, dtype=float)
            
            # Handle ground truth conversion more carefully
            ground_truth_processed = []
            for gt in ground_truth_list:
                if isinstance(gt, bool):
                    ground_truth_processed.append(1 if gt else 0)
                elif isinstance(gt, (int, float)):
                    ground_truth_processed.append(int(gt))
                elif gt is None:
                    continue  # Skip None values
                else:
                    # Try to convert to int
                    try:
                        ground_truth_processed.append(int(gt))
                    except (ValueError, TypeError):
                        continue  # Skip invalid values
            
            if len(ground_truth_processed) == 0:
                return {
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                    'auc': np.nan,
                    'confusion_matrix': None
                }
            
            binary_ground_truth = np.array(ground_truth_processed, dtype=int)
            
            # Ensure we have the same length for both arrays
            min_length = min(len(predictions_array), len(binary_ground_truth))
            predictions_array = predictions_array[:min_length]
            binary_ground_truth = binary_ground_truth[:min_length]
            
            # Convert predictions to binary using threshold
            binary_predictions = (predictions_array >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(binary_ground_truth, binary_predictions)
            precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
            recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
            f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
            
            # Calculate AUC (only if we have both classes)
            try:
                auc = roc_auc_score(binary_ground_truth, predictions_array)
            except ValueError:
                auc = np.nan  # Handle case where only one class is present
            
            # Confusion matrix
            cm = confusion_matrix(binary_ground_truth, binary_predictions)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm
            }
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            print(f"Predictions type: {type(predictions)}, length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
            print(f"Ground truth type: {type(ground_truth)}, length: {len(ground_truth) if hasattr(ground_truth, '__len__') else 'N/A'}")
            if hasattr(predictions, '__len__') and len(predictions) > 0:
                print(f"Predictions sample: {predictions[:3]}")
            if hasattr(ground_truth, '__len__') and len(ground_truth) > 0:
                print(f"Ground truth sample: {ground_truth[:3]}")
            return {
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan,
                'auc': np.nan,
                'confusion_matrix': None
            }
    
    def prepare_dataframe(self):
        """Convert JSON data to DataFrame with calculated ground truth averages."""
        if not self.all_data:
            self.load_all_data()
        
        records = []
        
        # Mapping of evaluation dimensions to ground truth dimensions
        dimension_mapping = {
            'audio_quality': 'audio_quality',
            'human_likeness': 'human_likeness',
            'appropriateness': 'appropriateness',
            'content_pass': 'content_pass',
        }
        
        for record in self.all_data:
            if not record.get('ground_truth_ratings'):
                continue
                
            # Get the specific dimension being evaluated
            eval_dimension = record.get('specific_dimension')
            ground_truth_dimension = dimension_mapping.get(eval_dimension, eval_dimension)
            
            # Calculate ground truth average for the specific dimension
            gt_ratings = record['ground_truth_ratings'].get(ground_truth_dimension, [])
            
            valid_ratings = self.parse_ratings(gt_ratings, eval_dimension)
            
            if not valid_ratings:
                continue
            
            # Handle content_pass differently - use majority vote instead of average
            if eval_dimension == "content_pass":
                gt_average = self.compute_majority_vote(valid_ratings)
                if gt_average is None:
                    continue  # Skip if no majority vote
            else:
                gt_average = np.mean(valid_ratings)
            
            # Calculate averages for all dimensions (for comprehensive analysis)
            all_averages = {}
            for dim, ratings in record['ground_truth_ratings'].items():
                if dim != 'annotator_confidence_rating':  # Exclude confidence rating
                    valid_ratings_dim = self.parse_ratings(ratings, dim)
                    if valid_ratings_dim:
                        try:
                            if dim == "content_pass":
                                # For content_pass, use majority vote instead of mean
                                majority_vote = self.compute_majority_vote(valid_ratings_dim)
                                all_averages[f'gt_avg_{dim}'] = majority_vote
                            else:
                                all_averages[f'gt_avg_{dim}'] = np.mean(valid_ratings_dim)
                        except Exception as e:
                            print(f"Error calculating average for {dim}: {e}")
                            print(f"valid_ratings_dim: {valid_ratings_dim}")
                            all_averages[f'gt_avg_{dim}'] = np.nan
            
            record_data = {
                'id': record.get('id'),
                'source_file': record.get('source_file'),
                'eval_dimension': eval_dimension,
                'model_score': record.get('model_score'),
                'probability_score': record.get('probability_score'),
                'gt_average': gt_average,
                'gt_count': len(valid_ratings),
                'character': record.get('char_profile', '')[:50] + '...' if record.get('char_profile') else '',
                **all_averages
            }
            
            records.append(record_data)
        
        self.df = pd.DataFrame(records)
        print(f"Prepared DataFrame with {len(self.df)} valid records")
        return self.df
    
    def debug_nan_values(self, predictions, ground_truth, dimension_name=""):
        """Debug function to find NaN values in predictions and ground truth."""
        print(f"\n=== NaN DEBUGGING for {dimension_name} ===")
        
        # Check predictions
        if isinstance(predictions, list):
            pred_nan_indices = [i for i, val in enumerate(predictions) if isinstance(val, (int, float)) and np.isnan(val)]
            pred_nan_count = len(pred_nan_indices)
            print(f"Predictions NaN count: {pred_nan_count}")
            if pred_nan_count > 0:
                print(f"Predictions NaN indices: {pred_nan_indices}")
                print(f"Predictions NaN values: {[predictions[i] for i in pred_nan_indices]}")
        else:
            pred_nan_mask = np.isnan(predictions)
            pred_nan_count = np.sum(pred_nan_mask)
            print(f"Predictions NaN count: {pred_nan_count}")
            if pred_nan_count > 0:
                pred_nan_indices = np.where(pred_nan_mask)[0]
                print(f"Predictions NaN indices: {pred_nan_indices.tolist()}")
                print(f"Predictions NaN values: {predictions[pred_nan_mask].tolist()}")
        
        # Check ground truth
        if isinstance(ground_truth, list):
            gt_nan_indices = [i for i, val in enumerate(ground_truth) if isinstance(val, (int, float)) and np.isnan(val)]
            gt_nan_count = len(gt_nan_indices)
            print(f"Ground truth NaN count: {gt_nan_count}")
            if gt_nan_count > 0:
                print(f"Ground truth NaN indices: {gt_nan_indices}")
                print(f"Ground truth NaN values: {[ground_truth[i] for i in gt_nan_indices]}")
        else:
            gt_nan_mask = np.isnan(ground_truth)
            gt_nan_count = np.sum(gt_nan_mask)
            print(f"Ground truth NaN count: {gt_nan_count}")
            if gt_nan_count > 0:
                gt_nan_indices = np.where(gt_nan_mask)[0]
                print(f"Ground truth NaN indices: {gt_nan_indices.tolist()}")
                print(f"Ground truth NaN values: {ground_truth[gt_nan_mask].tolist()}")
        
        # Check for infinite values
        if isinstance(predictions, list):
            pred_inf_indices = [i for i, val in enumerate(predictions) if isinstance(val, (int, float)) and np.isinf(val)]
            pred_inf_count = len(pred_inf_indices)
            print(f"Predictions Inf count: {pred_inf_count}")
            if pred_inf_count > 0:
                print(f"Predictions Inf indices: {pred_inf_indices}")
        else:
            pred_inf_mask = np.isinf(predictions)
            pred_inf_count = np.sum(pred_inf_mask)
            print(f"Predictions Inf count: {pred_inf_count}")
            if pred_inf_count > 0:
                pred_inf_indices = np.where(pred_inf_mask)[0]
                print(f"Predictions Inf indices: {pred_inf_indices.tolist()}")
        
        if isinstance(ground_truth, list):
            gt_inf_indices = [i for i, val in enumerate(ground_truth) if isinstance(val, (int, float)) and np.isinf(val)]
            gt_inf_count = len(gt_inf_indices)
            print(f"Ground truth Inf count: {gt_inf_count}")
            if gt_inf_count > 0:
                print(f"Ground truth Inf indices: {gt_inf_indices}")
        else:
            gt_inf_mask = np.isinf(ground_truth)
            gt_inf_count = np.sum(gt_inf_mask)
            print(f"Ground truth Inf count: {gt_inf_count}")
            if gt_inf_count > 0:
                gt_inf_indices = np.where(gt_inf_mask)[0]
                print(f"Ground truth Inf indices: {gt_inf_indices.tolist()}")
        
        # Check data types and shapes
        print(f"Predictions type: {type(predictions)}, length: {len(predictions)}")
        print(f"Ground truth type: {type(ground_truth)}, length: {len(ground_truth)}")
        
        # Show sample values
        if len(predictions) > 0:
            print(f"Predictions sample (first 5): {predictions[:5]}")
        if len(ground_truth) > 0:
            print(f"Ground truth sample (first 5): {ground_truth[:5]}")
        
        print("=" * 50)
        
        # Return summary
        return {
            'pred_nan_count': pred_nan_count,
            'gt_nan_count': gt_nan_count,
            'pred_inf_count': pred_inf_count,
            'gt_inf_count': gt_inf_count,
            'has_nan': pred_nan_count > 0 or gt_nan_count > 0,
            'has_inf': pred_inf_count > 0 or gt_inf_count > 0
        }

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate evaluation metrics with focus on correlation metrics."""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                'MAE': np.nan,
                'MSE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan,
                'Pearson_r': np.nan,
                'Pearson_p': np.nan,
                'Spearman_r': np.nan,
                'Spearman_p': np.nan
            }

        try:
            # Convert to lists first to handle pandas Series and mixed types
            if hasattr(predictions, 'tolist'):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)
            
            if hasattr(ground_truth, 'tolist'):
                ground_truth_list = ground_truth.tolist()
            else:
                ground_truth_list = list(ground_truth)
            
            # Convert to numpy arrays with explicit dtype handling
            predictions_array = np.array(predictions_list, dtype=float)
            ground_truth_array = np.array(ground_truth_list, dtype=float)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(predictions_array) & np.isfinite(ground_truth_array)
            if not np.any(valid_mask):
                return {
                    'MAE': np.nan,
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R²': np.nan,
                    'Pearson_r': np.nan,
                    'Pearson_p': np.nan,
                    'Spearman_r': np.nan,
                    'Spearman_p': np.nan
                }
            
            predictions_clean = predictions_array[valid_mask]
            ground_truth_clean = ground_truth_array[valid_mask]
            
            mae = mean_absolute_error(ground_truth_clean, predictions_clean)
            mse = mean_squared_error(ground_truth_clean, predictions_clean)
            rmse = np.sqrt(mse)
            r2 = r2_score(ground_truth_clean, predictions_clean)
            
            # Pearson correlation
            correlation, p_value = stats.pearsonr(predictions_clean, ground_truth_clean)
            
            # Spearman correlation (rank-based)
            spearman_corr, spearman_p = stats.spearmanr(predictions_clean, ground_truth_clean)
            
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
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            print(f"Predictions type: {type(predictions)}, length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
            print(f"Ground truth type: {type(ground_truth)}, length: {len(ground_truth) if hasattr(ground_truth, '__len__') else 'N/A'}")
            if hasattr(predictions, '__len__') and len(predictions) > 0:
                print(f"Predictions sample: {predictions[:3]}")
            if hasattr(ground_truth, '__len__') and len(ground_truth) > 0:
                print(f"Ground truth sample: {ground_truth[:3]}")
            return {
                'MAE': np.nan,
                'MSE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan,
                'Pearson_r': np.nan,
                'Pearson_p': np.nan,
                'Spearman_r': np.nan,
                'Spearman_p': np.nan
            }
    
    def evaluate_scores(self):
        """Main evaluation function with focus on correlation metrics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        if self.df.empty:
            print("No valid data to evaluate")
            return {}, {}
        
        print("=== MULTI-DIMENSION SCORE EVALUATION RESULTS ===\n")
        
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
        
        # Overall evaluation
        print("=== OVERALL EVALUATION (ALL DIMENSIONS) ===")
        overall_prob_metrics = self.calculate_metrics(self.df['probability_score'], self.df['gt_average'])
        overall_model_metrics = self.calculate_metrics(self.df['model_score'], self.df['gt_average'])

        
        print("Model Score vs Ground Truth:")
        print(f"  Pearson r: {overall_model_metrics['Pearson_r']:.4f} (p={overall_model_metrics['Pearson_p']:.4f})")
        print(f"  Spearman r: {overall_model_metrics['Spearman_r']:.4f} (p={overall_model_metrics['Spearman_p']:.4f})")
        print(f"  MAE: {overall_model_metrics['MAE']:.4f}")
        print(f"  RMSE: {overall_model_metrics['RMSE']:.4f}")
        
        print("\nProbability Score vs Ground Truth:")
        print(f"  Pearson r: {overall_prob_metrics['Pearson_r']:.4f} (p={overall_prob_metrics['Pearson_p']:.4f})")
        print(f"  Spearman r: {overall_prob_metrics['Spearman_r']:.4f} (p={overall_prob_metrics['Spearman_p']:.4f})")
        print(f"  MAE: {overall_prob_metrics['MAE']:.4f}")
        print(f"  RMSE: {overall_prob_metrics['RMSE']:.4f}")
        
        print("\nOverall Comparison:")
        print(f"Model Score Pearson r: {overall_model_metrics['Pearson_r']:.4f}")
        print(f"Probability Score Pearson r: {overall_prob_metrics['Pearson_r']:.4f}")
        print(f"Better Pearson: {'Model Score' if abs(overall_model_metrics['Pearson_r']) > abs(overall_prob_metrics['Pearson_r']) else 'Probability Score'}")
        print(f"Model Score Spearman r: {overall_model_metrics['Spearman_r']:.4f}")
        print(f"Probability Score Spearman r: {overall_prob_metrics['Spearman_r']:.4f}")
        print(f"Better Spearman: {'Model Score' if abs(overall_model_metrics['Spearman_r']) > abs(overall_prob_metrics['Spearman_r']) else 'Probability Score'}")
        print()
        
        # Dimension-specific analysis
        print("=== DIMENSION-SPECIFIC ANALYSIS ===")
        dimension_results = {}
        
        for dimension in self.df['eval_dimension'].unique():
            dim_data = self.df[self.df['eval_dimension'] == dimension]
            
            if len(dim_data) > 1:
                print(f"\n{dimension.upper()} (n={len(dim_data)}):")
                
                # Special handling for content_pass - use classification metrics
                if dimension == "content_pass":
                    # Calculate classification metrics for probability scores
                    prob_class_metrics = self.calculate_classification_metrics(
                        dim_data['probability_score'], 
                        dim_data['gt_average']
                    )
                    
                    # Calculate classification metrics for model scores (convert to 0-1 range)
                    model_scores_normalized = (dim_data['model_score'] - 1) / 4  # Convert 1-5 to 0-1
                    model_class_metrics = self.calculate_classification_metrics(
                        model_scores_normalized, 
                        dim_data['gt_average']
                    )
                    
                    dimension_results[dimension] = {
                        'model_class_metrics': model_class_metrics,
                        'prob_class_metrics': prob_class_metrics,
                        'count': len(dim_data)
                    }
                    
                    print(f"  Model Score Classification:")
                    print(f"    Accuracy: {model_class_metrics['accuracy']:.4f}")
                    print(f"    Precision: {model_class_metrics['precision']:.4f}")
                    print(f"    Recall: {model_class_metrics['recall']:.4f}")
                    print(f"    F1: {model_class_metrics['f1']:.4f}")
                    print(f"    AUC: {model_class_metrics['auc']:.4f}")
                    
                    print(f"  Probability Score Classification:")
                    print(f"    Accuracy: {prob_class_metrics['accuracy']:.4f}")
                    print(f"    Precision: {prob_class_metrics['precision']:.4f}")
                    print(f"    Recall: {prob_class_metrics['recall']:.4f}")
                    print(f"    F1: {prob_class_metrics['f1']:.4f}")
                    print(f"    AUC: {prob_class_metrics['auc']:.4f}")
                    
                    print(f"  Better Accuracy: {'Model' if model_class_metrics['accuracy'] > prob_class_metrics['accuracy'] else 'Probability'}")
                    print(f"  Better F1: {'Model' if model_class_metrics['f1'] > prob_class_metrics['f1'] else 'Probability'}")
                    print(f"  Better AUC: {'Model' if model_class_metrics['auc'] > prob_class_metrics['auc'] else 'Probability'}")
                    
                else:
                    # Regular regression metrics for other dimensions
                    dim_model_metrics = self.calculate_metrics(dim_data['model_score'], dim_data['gt_average'])
                    dim_prob_metrics = self.calculate_metrics(dim_data['probability_score'], dim_data['gt_average'])
                    
                    dimension_results[dimension] = {
                        'model_metrics': dim_model_metrics,
                        'prob_metrics': dim_prob_metrics,
                        'count': len(dim_data)
                    }
                    
                    print(f"  Model Score - Pearson r: {dim_model_metrics['Pearson_r']:.4f}, Spearman r: {dim_model_metrics['Spearman_r']:.4f}")
                    print(f"  Probability Score - Pearson r: {dim_prob_metrics['Pearson_r']:.4f}, Spearman r: {dim_prob_metrics['Spearman_r']:.4f}")
                    print(f"  Better Pearson: {'Model' if abs(dim_model_metrics['Pearson_r']) > abs(dim_prob_metrics['Pearson_r']) else 'Probability'}")
                    print(f"  Better Spearman: {'Model' if abs(dim_model_metrics['Spearman_r']) > abs(dim_prob_metrics['Spearman_r']) else 'Probability'}")
        
        return overall_model_metrics, overall_prob_metrics, dimension_results
    
    def create_visualizations(self, output_dir="."):
        """Create comprehensive visualizations with focus on correlation metrics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        if self.df.empty:
            print("No data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        
        # 1. Correlation heatmap
        dimensions = self.df['eval_dimension'].unique()
        
        # Prepare correlation data for heatmap
        correlation_data = []
        dimension_names = []
        
        for dimension in dimensions:
            dim_data = self.df[self.df['eval_dimension'] == dimension]
            if len(dim_data) > 1:
                dimension_names.append(dimension)
                
                # Calculate metrics for this dimension
                dim_model_metrics = self.calculate_metrics(dim_data['model_score'], dim_data['gt_average'])
                dim_prob_metrics = self.calculate_metrics(dim_data['probability_score'], dim_data['gt_average'])
                
                correlation_data.append([
                    dim_model_metrics['Pearson_r'],
                    dim_model_metrics['Spearman_r'],
                    dim_prob_metrics['Pearson_r'],
                    dim_prob_metrics['Spearman_r']
                ])
        
        if correlation_data:
            correlation_df = pd.DataFrame(
                correlation_data,
                index=dimension_names,
                columns=['Model_Pearson', 'Model_Spearman', 'Prob_Pearson', 'Prob_Spearman']
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_df, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f')
            plt.title('Correlation Metrics by Dimension')
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Correlation heatmap saved to: {heatmap_path}")
        
        # 2. Overall scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Overall Model vs Ground Truth Score Evaluation', fontsize=16, fontweight='bold')
        
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
        overall_plot_path = os.path.join(output_dir, 'overall_score_evaluation.png')
        plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overall evaluation plots saved to: {overall_plot_path}")
        
        # 3. Correlation comparison across dimensions
        if len(dimension_names) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prepare data for plotting
            model_pearson_scores = []
            prob_pearson_scores = []
            model_spearman_scores = []
            prob_spearman_scores = []
            
            for dimension in dimension_names:
                dim_data = self.df[self.df['eval_dimension'] == dimension]
                if len(dim_data) > 1:
                    # Calculate metrics
                    model_metrics = self.calculate_metrics(dim_data['model_score'], dim_data['gt_average'])
                    prob_metrics = self.calculate_metrics(dim_data['probability_score'], dim_data['gt_average'])
                    
                    model_pearson_scores.append(model_metrics['Pearson_r'])
                    prob_pearson_scores.append(prob_metrics['Pearson_r'])
                    model_spearman_scores.append(model_metrics['Spearman_r'])
                    prob_spearman_scores.append(prob_metrics['Spearman_r'])
            
            # Pearson correlation comparison
            x = np.arange(len(dimension_names))
            width = 0.35
            
            axes[0].bar(x - width/2, model_pearson_scores, width, label='Model Score', alpha=0.8, color='blue')
            axes[0].bar(x + width/2, prob_pearson_scores, width, label='Probability Score', alpha=0.8, color='green')
            axes[0].set_ylabel('Pearson Correlation')
            axes[0].set_title('Pearson Correlation by Dimension')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(dimension_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Spearman correlation comparison
            axes[1].bar(x - width/2, model_spearman_scores, width, label='Model Score', alpha=0.8, color='blue')
            axes[1].bar(x + width/2, prob_spearman_scores, width, label='Probability Score', alpha=0.8, color='green')
            axes[1].set_ylabel('Spearman Correlation')
            axes[1].set_title('Spearman Correlation by Dimension')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(dimension_names, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            correlation_comparison_path = os.path.join(output_dir, 'correlation_comparison.png')
            plt.savefig(correlation_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Correlation comparison plots saved to: {correlation_comparison_path}")
        
        # 4. Error metrics comparison
        if len(dimension_names) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prepare error data
            model_mae_scores = []
            prob_mae_scores = []
            model_rmse_scores = []
            prob_rmse_scores = []
            
            for dimension in dimension_names:
                dim_data = self.df[self.df['eval_dimension'] == dimension]
                if len(dim_data) > 1:
                    # Calculate metrics
                    model_metrics = self.calculate_metrics(dim_data['model_score'], dim_data['gt_average'])
                    prob_metrics = self.calculate_metrics(dim_data['probability_score'], dim_data['gt_average'])
                    
                    model_mae_scores.append(model_metrics['MAE'])
                    prob_mae_scores.append(prob_metrics['MAE'])
                    model_rmse_scores.append(model_metrics['RMSE'])
                    prob_rmse_scores.append(prob_metrics['RMSE'])
            
            # MAE comparison
            x = np.arange(len(dimension_names))
            width = 0.35
            
            axes[0].bar(x - width/2, model_mae_scores, width, label='Model Score', alpha=0.8, color='blue')
            axes[0].bar(x + width/2, prob_mae_scores, width, label='Probability Score', alpha=0.8, color='green')
            axes[0].set_ylabel('Mean Absolute Error')
            axes[0].set_title('MAE by Dimension')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(dimension_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # RMSE comparison
            axes[1].bar(x - width/2, model_rmse_scores, width, label='Model Score', alpha=0.8, color='blue')
            axes[1].bar(x + width/2, prob_rmse_scores, width, label='Probability Score', alpha=0.8, color='green')
            axes[1].set_ylabel('Root Mean Square Error')
            axes[1].set_title('RMSE by Dimension')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(dimension_names, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            error_metrics_path = os.path.join(output_dir, 'error_metrics.png')
            plt.savefig(error_metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Error metrics plots saved to: {error_metrics_path}")

    def export_detailed_results(self, output_dir=".", filename="detailed_results.csv"):
        """Export detailed results including dimension-specific metrics."""
        if self.df.empty:
            self.prepare_dataframe()
        
        # Export main dataframe
        main_csv_path = os.path.join(output_dir, filename)
        self.df.to_csv(main_csv_path, index=False)
        print(f"Main results exported to: {main_csv_path}")
        
        # Create summary statistics with focus on correlation metrics
        summary_data = []
        
        # Overall metrics
        overall_model_metrics = self.calculate_metrics(self.df['model_score'], self.df['gt_average'])
        overall_prob_metrics = self.calculate_metrics(self.df['probability_score'], self.df['gt_average'])
        
        summary_data.append({
            'dimension': 'OVERALL',
            'count': len(self.df),
            'model_pearson_r': overall_model_metrics['Pearson_r'],
            'model_spearman_r': overall_model_metrics['Spearman_r'],
            'model_mae': overall_model_metrics['MAE'],
            'model_rmse': overall_model_metrics['RMSE'],
            'prob_pearson_r': overall_prob_metrics['Pearson_r'],
            'prob_spearman_r': overall_prob_metrics['Spearman_r'],
            'prob_mae': overall_prob_metrics['MAE'],
            'prob_rmse': overall_prob_metrics['RMSE'],
            'better_pearson': 'Model' if abs(overall_model_metrics['Pearson_r']) > abs(overall_prob_metrics['Pearson_r']) else 'Probability',
            'better_spearman': 'Model' if abs(overall_model_metrics['Spearman_r']) > abs(overall_prob_metrics['Spearman_r']) else 'Probability',
            'better_mae': 'Model' if overall_model_metrics['MAE'] < overall_prob_metrics['MAE'] else 'Probability'
        })
        
        # Dimension-specific metrics
        for dimension in self.df['eval_dimension'].unique():
            dim_data = self.df[self.df['eval_dimension'] == dimension]
            
            if len(dim_data) > 1:
                if dimension == "content_pass":
                    # Classification metrics for content_pass
                    model_scores_normalized = (dim_data['model_score'] - 1) / 4
                    model_class_metrics = self.calculate_classification_metrics(
                        model_scores_normalized, dim_data['gt_average']
                    )
                    prob_class_metrics = self.calculate_classification_metrics(
                        dim_data['probability_score'], dim_data['gt_average']
                    )
                    
                    summary_data.append({
                        'dimension': dimension,
                        'count': len(dim_data),
                        'model_accuracy': model_class_metrics['accuracy'],
                        'model_precision': model_class_metrics['precision'],
                        'model_recall': model_class_metrics['recall'],
                        'model_f1': model_class_metrics['f1'],
                        'model_auc': model_class_metrics['auc'],
                        'prob_accuracy': prob_class_metrics['accuracy'],
                        'prob_precision': prob_class_metrics['precision'],
                        'prob_recall': prob_class_metrics['recall'],
                        'prob_f1': prob_class_metrics['f1'],
                        'prob_auc': prob_class_metrics['auc'],
                        'better_accuracy': 'Model' if model_class_metrics['accuracy'] > prob_class_metrics['accuracy'] else 'Probability',
                        'better_f1': 'Model' if model_class_metrics['f1'] > prob_class_metrics['f1'] else 'Probability',
                        'better_auc': 'Model' if model_class_metrics['auc'] > prob_class_metrics['auc'] else 'Probability'
                    })
                else:
                    # Regression metrics for other dimensions
                    dim_model_metrics = self.calculate_metrics(dim_data['model_score'], dim_data['gt_average'])
                    dim_prob_metrics = self.calculate_metrics(dim_data['probability_score'], dim_data['gt_average'])
                    
                    summary_data.append({
                        'dimension': dimension,
                        'count': len(dim_data),
                        'model_pearson_r': dim_model_metrics['Pearson_r'],
                        'model_spearman_r': dim_model_metrics['Spearman_r'],
                        'model_mae': dim_model_metrics['MAE'],
                        'model_rmse': dim_model_metrics['RMSE'],
                        'prob_pearson_r': dim_prob_metrics['Pearson_r'],
                        'prob_spearman_r': dim_prob_metrics['Spearman_r'],
                        'prob_mae': dim_prob_metrics['MAE'],
                        'prob_rmse': dim_prob_metrics['RMSE'],
                        'better_pearson': 'Model' if abs(dim_model_metrics['Pearson_r']) > abs(dim_prob_metrics['Pearson_r']) else 'Probability',
                        'better_spearman': 'Model' if abs(dim_model_metrics['Spearman_r']) > abs(dim_prob_metrics['Spearman_r']) else 'Probability',
                        'better_mae': 'Model' if dim_model_metrics['MAE'] < dim_prob_metrics['MAE'] else 'Probability'
                    })
        
        # Export summary
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary results exported to: {summary_csv_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model scores against ground truth ratings across multiple dimensions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing JSON files with evaluation results"
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
        help="Create detailed CSV outputs with dimension-specific metrics"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = MultiDimensionScoreEvaluator(args.folder)
    
    if not evaluator.json_files:
        print("No JSON files found to process.")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = args.folder
    
    # Run evaluation
    try:
        overall_model_metrics, overall_prob_metrics, dimension_results = evaluator.evaluate_scores()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualizations if not disabled
        if not args.no_plots:
            evaluator.create_visualizations(args.output_dir)
        
        # Export detailed results if requested
        if args.detailed_output:
            evaluator.export_detailed_results(args.output_dir)
        else:
            # Export basic results
            basic_csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
            evaluator.df.to_csv(basic_csv_path, index=False)
            print(f"Basic results exported to: {basic_csv_path}")
        
        print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
