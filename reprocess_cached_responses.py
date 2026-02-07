#!/usr/bin/env python3
"""
Reprocess Cached Responses - Efficient Pipeline Re-run Without API Calls

PURPOSE:
    Reprocess existing cached API responses with updated labeling logic and/or
    stratification, without making new expensive API calls.

USE CASES:
    - Fixed a bug in labeling logic → Re-label without re-querying API
    - Improved stratification → Re-split data properly
    - Updated feature extraction → Regenerate features from same responses
    - Changed model hyperparameters → Retrain with same features

WHAT IT DOES:
    1. Loads processed dataset CSV
    2. Loads cached response pickle files (_responses.pkl)
    3. Re-labels all responses using current labeling logic from ground_truth.py
    4. Creates stratified train/test split on actual hallucination_label
    5. Extracts features using current feature engineering
    6. Trains XGBoost model with proper stratification
    7. Calculates comprehensive metrics
    8. Saves outputs with "_reprocessed" suffix

COST SAVINGS:
    - No new API calls = $0 additional cost
    - Only CPU time (30-60 min for 500 samples)
    - Psychology example: 500 samples reprocessed in ~30 min, saved ~$5-10 in API costs

USAGE:
    python reprocess_cached_responses.py --domain psychology
    python reprocess_cached_responses.py --domain medicine

EXAMPLE SUCCESS (Psychology Stratification Fix):
    Before (broken): AUROC 0.5312, 26.5% train/test gap
    After (fixed):   AUROC 0.7115, 0.5% train/test gap (+34% improvement!)

OUTPUT FILES:
    data/features/{domain}_features_reprocessed.csv
    outputs/models/xgboost_{domain}_reprocessed.pkl
    outputs/results/metrics_{domain}_reprocessed.json

REQUIREMENTS:
    - Cached responses: data/features/{prompt_id}_responses.pkl
    - Processed dataset: data/processed/{domain}_processed.csv
    - Internet NOT required

NOTES:
    - Uses CURRENT labeling logic (any updates are applied automatically)
    - Does NOT overwrite original files (uses _reprocessed suffix)
    - Visualizations must be regenerated separately with generate_clean_viz.py

Created: February 2026 | Version: 1.0
"""

import argparse
import sys

# Add project to path
sys.path.insert(0, '/Users/aviv.gross/HHPF')

from src.data_preparation.label_responses import label_all_responses, save_labeled_responses
from src.features.feature_aggregator import FeatureAggregator
from src.classifier.xgboost_model import HallucinationClassifier
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualization import HHPFVisualizer
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Reprocess cached responses with proper stratification')
    parser.add_argument('--domain', type=str, required=True, help='Domain to reprocess')
    args = parser.parse_args()
    
    domain = args.domain
    
    logger.info("\n" + "="*80)
    logger.info(f"REPROCESSING CACHED RESPONSES - {domain.upper()}")
    logger.info("="*80)
    logger.info("\nThis will:")
    logger.info("  1. Use existing cached responses (no API calls)")
    logger.info("  2. Re-label with proper semantic similarity")
    logger.info("  3. Create stratified train/test split")
    logger.info("  4. Extract features")
    logger.info("  5. Train model")
    logger.info("\n" + "="*80)
    
    # Step 1: Label responses and create stratified split
    logger.info("\nSTEP 1: Labeling & Stratification")
    logger.info("="*80)
    
    processed_csv = f'data/processed/{domain}_processed.csv'
    
    if not Path(processed_csv).exists():
        logger.error(f"Processed CSV not found: {processed_csv}")
        logger.info("Run this first: python -m src.data_preparation.process_datasets --domain {domain}")
        sys.exit(1)
    
    try:
        labeled_df = label_all_responses(
            processed_csv=processed_csv,
            responses_dir='data/features',
            domain=domain,
            train_ratio=0.8,
            random_seed=42
        )
    except Exception as e:
        logger.error(f"Labeling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save labeled responses
    responses_path = f'data/features/responses_{domain}_reprocessed.csv'
    save_labeled_responses(labeled_df, responses_path)
    
    # Step 2: Feature extraction
    logger.info("\nSTEP 2: Feature Extraction")
    logger.info("="*80)
    
    try:
        aggregator = FeatureAggregator()
        features_df = aggregator.aggregate_features(
            responses_csv=responses_path,
            output_path=f'data/features/{domain}_features_reprocessed.csv'
        )
        
        features_path = f'data/features/{domain}_features_reprocessed.csv'
        logger.info(f"✓ Features extracted: {features_df.shape}")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Model training
    logger.info("\nSTEP 3: Model Training")
    logger.info("="*80)
    
    try:
        classifier = HallucinationClassifier()
        X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
        
        classifier.train(X_train, y_train)
        
        # Get predictions
        y_pred_test = classifier.predict(X_test)
        y_proba_test = classifier.predict_proba(X_test)
        
        # Calculate comprehensive metrics using MetricsCalculator
        calculator = MetricsCalculator()
        test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
        
        # Also get train metrics for completeness
        y_pred_train = classifier.predict(X_train)
        y_proba_train = classifier.predict_proba(X_train)
        train_metrics = calculator.calculate_all_metrics(y_train, y_pred_train, y_proba_train)
        
        # Save model
        model_path = f'outputs/models/xgboost_{domain}_reprocessed.pkl'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        classifier.save_model(model_path)
        
        logger.info(f"✓ Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Save metrics
    logger.info("\nSTEP 4: Saving Metrics")
    logger.info("="*80)
    
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
    
    metrics_path = f'outputs/results/metrics_{domain}_reprocessed.json'
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'train': convert_numpy(train_metrics),
            'test': convert_numpy(test_metrics)
        }, f, indent=2)
    
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    # Step 5: Visualizations
    logger.info("\nSTEP 5: Generating Visualizations")
    logger.info("="*80)
    
    figures_dir = f'outputs/figures/{domain}_reprocessed'
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    viz = HHPFVisualizer(output_dir=figures_dir)
    
    models_data = {'Full Model': test_metrics}
    viz.plot_roc_curves(models_data, filename=f'roc_curve_{domain}')
    viz.plot_arc(models_data, filename=f'arc_{domain}')
    viz.plot_calibration(test_metrics, filename=f'calibration_{domain}')
    viz.plot_confusion_matrix(test_metrics, filename=f'confusion_matrix_{domain}')
    
    importance_df = classifier.get_feature_importance(top_k=15)
    viz.plot_feature_importance(
        importance_df,
        title=f"Feature Importance - {domain.capitalize()} (Reprocessed)",
        filename=f'feature_importance_{domain}'
    )
    
    logger.info(f"✓ Figures saved to {figures_dir}/")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info(f"REPROCESSING COMPLETE - {domain.upper()}")
    logger.info("="*80)
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Test AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test ECE:      {test_metrics['ece']:.4f}")
    
    logger.info(f"\n📁 Files:")
    logger.info(f"  Features:  {features_path}")
    logger.info(f"  Model:     {model_path}")
    logger.info(f"  Metrics:   {metrics_path}")
    logger.info(f"  Figures:   {figures_dir}/")
    
    logger.info(f"\n✅ Ready for analysis!")


if __name__ == '__main__':
    main()
