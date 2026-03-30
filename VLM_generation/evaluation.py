"""
Evaluation: Qualitatively and quantitatively compare isolated-instance outputs
against a baseline using raw, unmasked images.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import os
from collections import defaultdict


def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable types to Python native types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class VLMEvaluator:
    """
    Evaluator for comparing VLM outputs on isolated instances vs. raw images.
    """
    
    def __init__(self):
        self.results = {
            'isolated': {},
            'baseline': {},
            'comparisons': []
        }
    
    def evaluate_qualitative(
        self,
        isolated_responses: Dict,
        baseline_responses: Dict,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Perform qualitative evaluation by comparing responses.
        
        Args:
            isolated_responses: Dictionary of responses from isolated instances
            baseline_responses: Dictionary of responses from raw images
            output_dir: Directory to save comparison results
        
        Returns:
            Dictionary with qualitative comparison results
        """
        comparisons = []
        
        # Match responses by key (class_idx, instance_idx)
        all_keys = set(isolated_responses.keys()) | set(baseline_responses.keys())
        
        for key in all_keys:
            isolated = isolated_responses.get(key, {}).get('response', 'N/A')
            baseline = baseline_responses.get(key, {}).get('response', 'N/A')
            
            # Convert key to serializable format
            serializable_key = convert_to_serializable(key)
            
            comparison = {
                'key': serializable_key,
                'isolated_response': isolated,
                'baseline_response': baseline,
                'class_name': isolated_responses.get(key, {}).get('class_name', 'unknown')
            }
            
            # Simple similarity metrics
            comparison['length_ratio'] = float(len(isolated) / max(len(baseline), 1))
            comparison['word_overlap'] = float(self._calculate_word_overlap(isolated, baseline))
            
            comparisons.append(comparison)
        
        self.results['comparisons'] = comparisons
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'qualitative_comparison.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons, f, indent=2, ensure_ascii=False)
            print(f"Qualitative comparison saved to {output_file}")
        
        return {
            'comparisons': comparisons,
            'num_comparisons': len(comparisons)
        }
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_quantitative(
        self,
        isolated_responses: Dict,
        baseline_responses: Dict,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform quantitative evaluation using various metrics.
        
        Args:
            isolated_responses: Dictionary of responses from isolated instances
            baseline_responses: Dictionary of responses from raw images
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary with quantitative metrics
        """
        if metrics is None:
            metrics = ['length', 'word_overlap', 'specificity', 'relevance']
        
        quantitative_results = {
            'metrics': {},
            'per_class': defaultdict(dict),
            'overall': {}
        }
        
        all_keys = set(isolated_responses.keys()) | set(baseline_responses.keys())
        
        # Collect metrics for each comparison
        metric_values = {metric: [] for metric in metrics}
        
        for key in all_keys:
            isolated = isolated_responses.get(key, {}).get('response', '')
            baseline = baseline_responses.get(key, {}).get('response', '')
            class_name = isolated_responses.get(key, {}).get('class_name', 'unknown')
            
            if 'length' in metrics:
                length_ratio = len(isolated) / max(len(baseline), 1)
                metric_values['length'].append(length_ratio)
                quantitative_results['per_class'][class_name]['length_ratio'] =                     quantitative_results['per_class'][class_name].get('length_ratio', []) + [length_ratio]
            
            if 'word_overlap' in metrics:
                overlap = self._calculate_word_overlap(isolated, baseline)
                metric_values['word_overlap'].append(overlap)
                quantitative_results['per_class'][class_name]['word_overlap'] =                     quantitative_results['per_class'][class_name].get('word_overlap', []) + [overlap]
            
            if 'specificity' in metrics:
                specificity = self._calculate_specificity(isolated, baseline)
                metric_values['specificity'].append(specificity)
                quantitative_results['per_class'][class_name]['specificity'] =                     quantitative_results['per_class'][class_name].get('specificity', []) + [specificity]
            
            if 'relevance' in metrics:
                relevance = self._calculate_relevance(isolated, class_name)
                metric_values['relevance'].append(relevance)
                quantitative_results['per_class'][class_name]['relevance'] =                     quantitative_results['per_class'][class_name].get('relevance', []) + [relevance]
        
        # Calculate overall statistics
        for metric in metrics:
            if metric_values[metric]:
                values = metric_values[metric]
                quantitative_results['overall'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Calculate per-class averages
        for class_name in quantitative_results['per_class']:
            for metric in quantitative_results['per_class'][class_name]:
                values = quantitative_results['per_class'][class_name][metric]
                quantitative_results['per_class'][class_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return quantitative_results
    
    def _calculate_specificity(self, isolated: str, baseline: str) -> float:
        """
        Calculate specificity: how much more specific/detailed is isolated vs baseline.
        Higher values indicate isolated instance produces more specific descriptions.
        """
        # Simple heuristic: compare length and unique words
        isolated_words = set(isolated.lower().split())
        baseline_words = set(baseline.lower().split())
        
        # More unique words in isolated = more specific
        isolated_unique = isolated_words - baseline_words
        baseline_unique = baseline_words - isolated_words
        
        if len(baseline_words) == 0:
            return 1.0 if len(isolated_words) > 0 else 0.0
        
        specificity = len(isolated_unique) / max(len(baseline_words), 1)
        return min(specificity, 1.0)  # Cap at 1.0
    
    def _calculate_relevance(self, response: str, class_name: str) -> float:
        """
        Calculate relevance: how relevant is the response to the class.
        Simple keyword-based relevance score.
        """
        response_lower = response.lower()
        class_lower = class_name.lower()
        
        # Check if class name appears in response
        if class_lower in response_lower:
            return 1.0
        
        # Check for related keywords (simple heuristic)
        related_keywords = {
            'car': ['vehicle', 'automobile', 'car'],
            'pedestrian': ['person', 'pedestrian', 'walking', 'human'],
            'bicyclist': ['bicycle', 'bike', 'cyclist', 'riding'],
            'truck': ['truck', 'vehicle', 'large'],
            'building': ['building', 'structure', 'construction'],
            'tree': ['tree', 'vegetation', 'plant'],
            'road': ['road', 'street', 'pavement', 'asphalt']
        }
        
        for key, keywords in related_keywords.items():
            if key in class_lower:
                for keyword in keywords:
                    if keyword in response_lower:
                        return 0.8
        
        return 0.5  # Default relevance
    
    def generate_comparison_report(
        self,
        output_dir: str,
        isolated_responses: Dict,
        baseline_responses: Dict
    ):
        """
        Generate a comprehensive comparison report.
        
        Args:
            output_dir: Directory to save the report
            isolated_responses: Responses from isolated instances
            baseline_responses: Responses from raw images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Qualitative evaluation
        qualitative = self.evaluate_qualitative(
            isolated_responses, baseline_responses, output_dir
        )
        
        # Quantitative evaluation
        quantitative = self.evaluate_quantitative(
            isolated_responses, baseline_responses
        )
        
        # Generate summary report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VLM Evaluation Report: Isolated Instances vs. Baseline\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Qualitative Evaluation:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Comparisons: {qualitative['num_comparisons']}\n\n")
            
            f.write("Quantitative Evaluation:\n")
            f.write("-" * 80 + "\n")
            f.write("Overall Metrics:\n")
            for metric, stats in quantitative['overall'].items():
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.4f}\n")
                f.write(f"    Std:  {stats['std']:.4f}\n")
                f.write(f"    Min:  {stats['min']:.4f}\n")
                f.write(f"    Max:  {stats['max']:.4f}\n")
                f.write(f"    Median: {stats['median']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            for class_name, metrics in quantitative['per_class'].items():
                f.write(f"  {class_name}:\n")
                for metric, stats in metrics.items():
                    f.write(f"    {metric}: mean={stats['mean']:.4f}, "
                           f"std={stats['std']:.4f}, count={stats['count']}\n")
                f.write("\n")
        
        # Save quantitative results as JSON
        json_path = os.path.join(output_dir, 'quantitative_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert to serializable format before dumping
            serializable_quantitative = convert_to_serializable(quantitative)
            json.dump(serializable_quantitative, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation report saved to {report_path}")
        print(f"Quantitative results saved to {json_path}")
