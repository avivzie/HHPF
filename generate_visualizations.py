#!/usr/bin/env python3
"""
Generate visualizations for reprocessed results.
Uses existing metrics and model - no re-training needed.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from src.classifier.xgboost_model import HallucinationClassifier
from src.evaluation.visualization import HHPFVisualizer
from src.evaluation.metrics import MetricsCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for reprocessed domain')
    parser.add_argument('--domain', type=str, required=True, help='Domain name')
    parser.add_argument('--suffix', type=str, default='reprocessed', help='File suffix (e.g., reprocessed)')
    args = parser.parse_args()
    
    domain = args.domain
    suffix = args.suffix
    
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING VISUALIZATIONS - {domain.upper()} ({suffix})")
    logger.info(f"{'='*80}\n")
    
    # Load model
    model_path = f'outputs/models/xgboost_{domain}_{suffix}.pkl'
    features_path = f'data/features/{domain}_features_{suffix}.csv'
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    if not Path(features_path).exists():
        logger.error(f"Features not found: {features_path}")
        return
    
    # Load classifier and data
    classifier = HallucinationClassifier()
    classifier.load_model(model_path)
    logger.info(f"✓ Loaded model from {model_path}")
    
    features_df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
    
    # Get predictions
    y_pred_test = classifier.predict(X_test)
    y_proba_test = classifier.predict_proba(X_test)
    
    # Calculate metrics with ROC curve data
    calculator = MetricsCalculator()
    test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
    
    logger.info(f"\n📊 Performance:")
    logger.info(f"  AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Verify stratification
    train_hall = (y_train == 1).sum() / len(y_train) * 100
    test_hall = (y_test == 1).sum() / len(y_test) * 100
    gap = abs(train_hall - test_hall)
    
    logger.info(f"\n📊 Stratification:")
    logger.info(f"  Train: {train_hall:.1f}% hallucinations")
    logger.info(f"  Test:  {test_hall:.1f}% hallucinations")
    logger.info(f"  Gap:   {gap:.1f}%")
    
    if gap < 5.0:
        logger.info(f"  ✅ Proper stratification (gap < 5%)")
    else:
        logger.warning(f"  ⚠️  Large gap: {gap:.1f}%")
    
    # Create visualizations
    logger.info(f"\nGenerating visualizations...")
    figures_dir = f'outputs/figures/{domain}_{suffix}'
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    viz = HHPFVisualizer(output_dir=figures_dir)
    
    models_data = {'Full Model': test_metrics}
    viz.plot_roc_curves(models_data, filename=f'roc_curve_{domain}')
    viz.plot_arc(models_data, filename=f'arc_{domain}')
    viz.plot_calibration(test_metrics, filename=f'calibration_{domain}')
    viz.plot_confusion_matrix(test_metrics, filename=f'confusion_matrix_{domain}')
    
    # Feature importance
    importance_df = classifier.get_feature_importance(top_k=15)
    viz.plot_feature_importance(
        importance_df,
        title=f"Feature Importance - {domain.capitalize()} (Reprocessed)",
        filename=f'feature_importance_{domain}'
    )
    
    logger.info(f"✓ Saved all figures to {figures_dir}/")
    
    # Save metrics to JSON
    metrics_path = f'outputs/results/metrics_{domain}_{suffix}.json'
    
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(metrics_path, 'w') as f:
        json.dump({'test': convert_numpy(test_metrics)}, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_path}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE - {domain.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"\nFiles generated:")
    logger.info(f"  Figures:  {figures_dir}/")
    logger.info(f"  Metrics:  {metrics_path}")
    logger.info(f"  Model:    {model_path}")
    logger.info(f"  Features: {features_path}")
    logger.info(f"\n✅ Domain ready for analysis!")


if __name__ == '__main__':
    main()
