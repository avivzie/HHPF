#!/usr/bin/env python3
"""
Generate Clean Visualizations - Regenerate Plots Without Retraining

PURPOSE:
    Regenerate visualizations from existing models and metrics without
    retraining or reprocessing. Useful for:
    - Fixing visualization bugs (e.g., matplotlib crashes)
    - Updating plot styles or formats
    - Regenerating after cleaning up output directories
    - Creating publication-ready figures

USE CASES:
    - Matplotlib crashed during pipeline run → Regenerate just the plots
    - Want different plot format (PDF vs PNG) → Regenerate with new settings
    - Cleaned up output directory → Restore visualizations
    - Need to update plot styling → Regenerate without full pipeline

WHAT IT DOES:
    1. Loads existing trained model (*_reprocessed.pkl)
    2. Loads existing feature matrix (*_features_reprocessed.csv)
    3. Calculates predictions and metrics
    4. Verifies stratification quality
    5. Generates all 5 standard visualizations:
       - ROC Curve (with AUROC)
       - ARC (Accuracy-Rejection Curve)
       - Calibration Plot (with ECE)
       - Confusion Matrix
       - Feature Importance (top 15 features)
    6. Saves in standard location: outputs/figures/{domain}/

SPEED:
    - Very fast: 30-60 seconds (no training, no API calls)
    - Only loads existing files and generates plots

USAGE:
    python generate_clean_viz.py --domain psychology
    python generate_clean_viz.py --domain medicine

OUTPUT FILES (5 plots × 2 formats = 10 files):
    outputs/figures/{domain}/roc_curve_{domain}.{pdf,png}
    outputs/figures/{domain}/arc_{domain}.{pdf,png}
    outputs/figures/{domain}/calibration_{domain}.{pdf,png}
    outputs/figures/{domain}/confusion_matrix_{domain}.{pdf,png}
    outputs/figures/{domain}/feature_importance_{domain}.{pdf,png}

ALSO SAVES:
    outputs/results/metrics_{domain}.json (comprehensive metrics)

REQUIREMENTS:
    - Trained model: outputs/models/xgboost_{domain}_reprocessed.pkl
    - Features: data/features/{domain}_features_reprocessed.csv
    - Internet NOT required

TECHNICAL NOTES:
    - Uses Agg backend (non-interactive) to avoid matplotlib crashes in sandbox
    - Generates both PDF (vector) and PNG (raster) formats
    - Saves to STANDARD location (outputs/figures/{domain}/) not temp directories
    - Verifies stratification and reports train/test gap

EXAMPLE OUTPUT:
    Psychology:
      AUROC: 0.7115
      Accuracy: 0.78
      Stratification gap: 0.5% (perfect!)
      All 10 files generated successfully

Created: February 2026 | Version: 1.0
"""

# Set non-interactive backend BEFORE importing matplotlib
import matplotlib
matplotlib.use('Agg')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True)
    args = parser.parse_args()
    
    domain = args.domain
    
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING CLEAN VISUALIZATIONS - {domain.upper()}")
    logger.info(f"{'='*80}\n")
    
    # Use reprocessed files
    model_path = f'outputs/models/xgboost_{domain}_reprocessed.pkl'
    features_path = f'data/features/{domain}_features_reprocessed.csv'
    
    # Output to standard location
    output_dir = f'outputs/figures/{domain}'
    metrics_path = f'outputs/results/metrics_{domain}.json'
    
    # Load model and data
    classifier = HallucinationClassifier()
    classifier.load_model(model_path)
    logger.info(f"✓ Loaded model")
    
    features_df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
    logger.info(f"✓ Loaded features: {len(features_df)} samples")
    
    # Get predictions
    y_pred_test = classifier.predict(X_test)
    y_proba_test = classifier.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    calculator = MetricsCalculator()
    test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
    
    # Verify stratification
    train_hall = (y_train == 1).sum() / len(y_train) * 100
    test_hall = (y_test == 1).sum() / len(y_test) * 100
    gap = abs(train_hall - test_hall)
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Train:    {train_hall:.1f}% hallucinations")
    logger.info(f"  Test:     {test_hall:.1f}% hallucinations")
    logger.info(f"  Gap:      {gap:.1f}%")
    
    # Save metrics to standard location
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    def convert_numpy(obj):
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
    
    # Generate visualizations
    logger.info(f"\nGenerating visualizations...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    viz = HHPFVisualizer(output_dir=output_dir)
    
    models_data = {'Full Model': test_metrics}
    viz.plot_roc_curves(models_data, filename=f'roc_curve_{domain}')
    viz.plot_arc(models_data, filename=f'arc_{domain}')
    viz.plot_calibration(test_metrics, filename=f'calibration_{domain}')
    viz.plot_confusion_matrix(test_metrics, filename=f'confusion_matrix_{domain}')
    
    # Feature importance
    importance_df = classifier.get_feature_importance(top_k=15)
    viz.plot_feature_importance(
        importance_df,
        title=f"Feature Importance - {domain.capitalize()}",
        filename=f'feature_importance_{domain}'
    )
    
    logger.info(f"✓ Saved all visualizations to {output_dir}/")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ {domain.upper()} VISUALIZATIONS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nFiles:")
    logger.info(f"  Figures: {output_dir}/")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"\n📊 Final Results:")
    logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"  Stratification gap: {gap:.1f}%")


if __name__ == '__main__':
    main()
