"""
Evaluation metrics for HHPF.

Implements:
1. AUROC (Area Under ROC Curve)
2. ARC (Accuracy-Rejection Curve)
3. ECE (Expected Calibration Error)
4. Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate research-grade evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_auroc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, any]:
        """
        Calculate AUROC and ROC curve data.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with AUROC score and ROC curve data
        """
        auroc = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        return {
            'auroc': auroc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def calculate_arc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        num_points: int = 20
    ) -> Dict[str, any]:
        """
        Calculate Accuracy-Rejection Curve.
        
        Shows how accuracy improves when rejecting uncertain predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            num_points: Number of points on the curve
            
        Returns:
            Dictionary with rejection rates and accuracies
        """
        # Sort by confidence (distance from 0.5)
        confidence = np.abs(y_proba - 0.5)
        sorted_indices = np.argsort(confidence)[::-1]  # Most confident first
        
        y_true_sorted = np.array(y_true)[sorted_indices]
        y_proba_sorted = np.array(y_proba)[sorted_indices]
        
        # Calculate accuracy at different rejection rates
        rejection_rates = np.linspace(0, 0.5, num_points)
        accuracies = []
        num_samples = len(y_true)
        
        for rejection_rate in rejection_rates:
            # Keep most confident (1 - rejection_rate) samples
            keep_ratio = 1.0 - rejection_rate
            n_keep = int(num_samples * keep_ratio)
            
            if n_keep == 0:
                accuracies.append(np.nan)
                continue
            
            # Calculate accuracy on kept samples
            y_true_kept = y_true_sorted[:n_keep]
            y_proba_kept = y_proba_sorted[:n_keep]
            y_pred_kept = (y_proba_kept >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_true_kept, y_pred_kept)
            accuracies.append(accuracy)
        
        return {
            'rejection_rates': rejection_rates,
            'accuracies': np.array(accuracies),
            'auc_arc': np.trapezoid(accuracies, rejection_rates)  # Area under ARC
        }
    
    def calculate_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, any]:
        """
        Calculate Expected Calibration Error.
        
        Measures how well predicted probabilities match actual outcomes.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with ECE and calibration data
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if not np.any(in_bin):
                bin_accuracies.append(np.nan)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
                continue
            
            # Accuracy of predictions in this bin
            bin_accuracy = y_true[in_bin].mean()
            
            # Average confidence in this bin
            bin_confidence = y_proba[in_bin].mean()
            
            # Number of samples in bin
            bin_count = np.sum(in_bin)
            
            # ECE contribution (weighted by bin size)
            ece += (bin_count / len(y_true)) * np.abs(bin_accuracy - bin_confidence)
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        
        return {
            'ece': ece,
            'bin_boundaries': bin_boundaries,
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts)
        }
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, any]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUROC
        auroc_data = self.calculate_auroc(y_true, y_proba)
        metrics['auroc'] = auroc_data['auroc']
        metrics['roc_curve'] = {
            'fpr': auroc_data['fpr'],
            'tpr': auroc_data['tpr'],
            'thresholds': auroc_data['thresholds']
        }
        
        # ARC
        arc_data = self.calculate_arc(y_true, y_proba)
        metrics['arc'] = {
            'rejection_rates': arc_data['rejection_rates'],
            'accuracies': arc_data['accuracies'],
            'auc_arc': arc_data['auc_arc']
        }
        
        # ECE
        ece_data = self.calculate_ece(y_true, y_proba)
        metrics['ece'] = ece_data['ece']
        metrics['calibration'] = {
            'bin_boundaries': ece_data['bin_boundaries'],
            'bin_accuracies': ece_data['bin_accuracies'],
            'bin_confidences': ece_data['bin_confidences'],
            'bin_counts': ece_data['bin_counts']
        }
        
        return metrics
    
    def compare_models(
        self,
        models_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models_results: Dictionary mapping model name to metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for model_name, metrics in models_results.items():
            row = {
                'model': model_name,
                'auroc': metrics.get('auroc', np.nan),
                'accuracy': metrics.get('accuracy', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
                'f1': metrics.get('f1', np.nan),
                'ece': metrics.get('ece', np.nan),
            }
            
            if 'arc' in metrics:
                row['auc_arc'] = metrics['arc'].get('auc_arc', np.nan)
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('auroc', ascending=False)
        
        return comparison_df


class FeatureImportanceAnalyzer:
    """Analyze feature importance and correlations."""
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze_shap_importance(
        self,
        model,
        X: pd.DataFrame,
        feature_names: List[str],
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate SHAP feature importance.
        
        Args:
            model: Trained XGBoost model
            X: Features
            feature_names: Feature names
            max_samples: Maximum samples for SHAP (computational efficiency)
            
        Returns:
            DataFrame with SHAP importance values
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return None
        
        # Subsample if needed
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Aggregate importance (mean absolute SHAP value)
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance
        })
        
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        
        return importance_df
    
    def calculate_feature_correlations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation between features and target.
        
        Args:
            X: Features
            y: Target variable
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            DataFrame with correlations
        """
        correlations = []
        
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                if method == 'pearson':
                    corr = X[col].corr(y)
                elif method == 'spearman':
                    corr = X[col].corr(y, method='spearman')
                else:
                    corr = np.nan
                
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        return corr_df
    
    def analyze_domain_specific_importance(
        self,
        features_df: pd.DataFrame,
        models: Dict[str, any],
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Analyze feature importance across domains.
        
        Args:
            features_df: Features with domain information
            models: Dictionary mapping domain to trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with domain-specific importance
        """
        importance_data = []
        
        for domain, model in models.items():
            # Get feature importance from model
            importance = model.feature_importances_
            
            for feat, imp in zip(feature_names, importance):
                importance_data.append({
                    'domain': domain,
                    'feature': feat,
                    'importance': imp
                })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Pivot to get domain comparison
        importance_pivot = importance_df.pivot(
            index='feature',
            columns='domain',
            values='importance'
        )
        
        return importance_pivot


def main():
    """Main entry point for metrics calculation."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description="Calculate HHPF evaluation metrics")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature matrix CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results/metrics.json',
        help='Output path for metrics'
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    from src.classifier.xgboost_model import HallucinationClassifier
    
    classifier = HallucinationClassifier()
    classifier.load_model(args.model)
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
    
    # Predictions
    y_pred_train = classifier.predict(X_train)
    y_proba_train = classifier.predict_proba(X_train)
    
    y_pred_test = classifier.predict(X_test)
    y_proba_test = classifier.predict_proba(X_test)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    
    logger.info("\nCalculating train set metrics...")
    train_metrics = calculator.calculate_all_metrics(y_train, y_pred_train, y_proba_train)
    
    logger.info("Calculating test set metrics...")
    test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    for set_name, metrics in [("Train", train_metrics), ("Test", test_metrics)]:
        logger.info(f"\n{set_name} Set:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:    {metrics['f1']:.4f}")
        logger.info(f"  AUROC:       {metrics['auroc']:.4f}")
        logger.info(f"  ECE:         {metrics['ece']:.4f}")
        logger.info(f"  AUC-ARC:     {metrics['arc']['auc_arc']:.4f}")
    
    # Save metrics
    import json
    from pathlib import Path
    from src.utils import ensure_dir
    
    ensure_dir(Path(args.output).parent)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    results = {
        'train': convert_numpy(train_metrics),
        'test': convert_numpy(test_metrics)
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
