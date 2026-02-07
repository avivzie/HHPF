#!/usr/bin/env python3
"""
Quick fix: Reprocess cached responses with proper stratification.

This script:
1. Loads all cached response files for a domain
2. Re-labels all responses consistently
3. Splits train/test with PROPER stratification on hallucination_label
4. Saves corrected processed CSV
5. Runs feature extraction and model training

Usage:
    python reprocess_with_stratification.py --domain medicine
    python reprocess_with_stratification.py --domain psychology
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.data_preparation.ground_truth import get_labeler
from src.features.feature_aggregator import FeatureAggregator
from src.classifier.xgboost_model import HallucinationClassifier
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualization import HHPFVisualizer
import json


def load_and_label_responses(domain: str, processed_csv_path: str, labeler):
    """Load cached responses and label them."""
    # Load processed data (has prompts and ground truth)
    df = pd.read_csv(processed_csv_path)
    logger.info(f"Loaded {len(df)} samples from {processed_csv_path}")
    
    cache_dir = Path('data/features')
    all_data = []
    
    for idx, row in df.iterrows():
        prompt_id = row['prompt_id']
        response_file = cache_dir / f"{prompt_id}_responses.pkl"
        
        if not response_file.exists():
            logger.warning(f"Missing response file for {prompt_id}, skipping")
            continue
        
        try:
            # Load responses (list of 5 stochastic responses)
            with open(response_file, 'rb') as f:
                responses = pickle.load(f)
            
            # Use first response as primary
            primary_response = responses[0]['text'] if isinstance(responses[0], dict) else responses[0]
            
            # Label the response
            label_result = labeler.label_response(
                response=primary_response,
                ground_truth=row['ground_truth'],
                domain=domain,
                prompt=row['prompt']
            )
            
            # Create record with all info
            record = {
                'prompt_id': prompt_id,
                'domain': row['domain'],
                'prompt': row['prompt'],
                'ground_truth': row['ground_truth'],
                'primary_response': primary_response,
                'hallucination_label': label_result['hallucination_label'],
                'label_confidence': label_result.get('confidence', 0.0),
                'label_method': label_result.get('method', 'unknown'),
                'response_file': str(response_file)
            }
            
            # Add domain-specific label fields
            for key, value in label_result.items():
                if key not in ['hallucination_label', 'confidence', 'method']:
                    record[key] = value
            
            all_data.append(record)
            
        except Exception as e:
            logger.warning(f"Failed to process {prompt_id}: {e}")
            continue
    
    logger.info(f"✓ Loaded and labeled {len(all_data)} samples")
    return all_data


def stratified_split(df, test_size=0.2, random_state=42):
    """Perform stratified train/test split on hallucination_label."""
    logger.info(f"\nPerforming stratified split (test_size={test_size})...")
    
    # Check if we have both classes
    label_counts = df['hallucination_label'].value_counts()
    logger.info(f"Label distribution before split:")
    logger.info(f"  Hallucinations: {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(df) * 100:.1f}%)")
    logger.info(f"  Faithful: {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(df) * 100:.1f}%)")
    
    if len(label_counts) < 2:
        raise ValueError("Cannot stratify - only one class present!")
    
    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['hallucination_label']  # ✅ PROPER STRATIFICATION
    )
    
    # Add split column
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Verify stratification worked
    logger.info(f"\n✓ Stratified split complete:")
    logger.info(f"  Train: {len(train_df)} samples")
    train_hall_pct = (train_df['hallucination_label'] == 1).sum() / len(train_df) * 100
    logger.info(f"    Hallucinations: {train_hall_pct:.1f}%")
    
    logger.info(f"  Test: {len(test_df)} samples")
    test_hall_pct = (test_df['hallucination_label'] == 1).sum() / len(test_df) * 100
    logger.info(f"    Hallucinations: {test_hall_pct:.1f}%")
    
    gap = abs(train_hall_pct - test_hall_pct)
    logger.info(f"  Train/Test Gap: {gap:.1f} percentage points")
    
    if gap > 5.0:
        logger.warning(f"⚠️  Train/test gap is {gap:.1f}% (should be <5%)")
    else:
        logger.info(f"  ✓ Gap is acceptable (<5%)")
    
    return pd.concat([train_df, test_df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Reprocess domain with proper stratification')
    parser.add_argument('--domain', type=str, required=True, help='Domain to reprocess')
    parser.add_argument('--limit', type=int, default=500, help='Number of samples to use')
    args = parser.parse_args()
    
    domain = args.domain
    logger.info(f"\n{'='*80}")
    logger.info(f"REPROCESSING {domain.upper()} WITH PROPER STRATIFICATION")
    logger.info(f"{'='*80}\n")
    
    # Step 1: Load cached responses and label them
    logger.info("STEP 1: Loading cached responses and labeling")
    
    # Load the processed CSV (which has prompts and ground truth but no labels)
    processed_csv = f'data/processed/{domain}_processed.csv'
    labeler = get_labeler(domain)
    
    # Load responses and label them
    labeled_data = load_and_label_responses(domain, processed_csv, labeler)
    
    # Limit to requested number
    if len(labeled_data) > args.limit:
        logger.info(f"Limiting to {args.limit} samples (have {len(labeled_data)})")
        labeled_data = labeled_data[:args.limit]
    
    # Convert to DataFrame
    df = pd.DataFrame(labeled_data)
    
    # Step 2: Stratified split
    logger.info(f"\nSTEP 2: Stratified train/test split")
    df = stratified_split(df, test_size=0.2, random_state=42)
    
    # Step 3: Save corrected processed CSV
    output_file = f'data/processed/{domain}_processed_stratified.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved corrected data to {output_file}")
    
    # Step 4: Feature extraction
    logger.info(f"\nSTEP 3: Feature extraction")
    logger.info(f"{'='*80}")
    
    # Save responses for feature extraction
    responses_file = f'data/features/{domain}_responses_stratified.csv'
    df.to_csv(responses_file, index=False)
    
    # Extract features
    aggregator = FeatureAggregator()
    features_df = aggregator.aggregate_features(responses_file, domain)
    
    # Save features
    features_file = f'data/features/{domain}_features_stratified.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"✓ Saved features to {features_file}")
    
    # Step 5: Model training
    logger.info(f"\nSTEP 4: Model training")
    logger.info(f"{'='*80}")
    
    classifier = HallucinationClassifier()
    test_auroc = classifier.train(features_file, domain)
    
    logger.info(f"\n✓ Model trained - Test AUROC: {test_auroc:.4f}")
    
    # Step 6: Evaluation & Visualization
    logger.info(f"\nSTEP 5: Evaluation & Visualization")
    logger.info(f"{'='*80}")
    
    # Load model and prepare data
    model_path = f'outputs/models/xgboost_{domain}.pkl'
    classifier.load_model(model_path)
    
    features_full_df = pd.read_csv(features_file)
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_full_df)
    
    # Predictions
    y_pred_test = classifier.predict(X_test)
    y_proba_test = classifier.predict_proba(X_test)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
    
    # Save metrics
    metrics_file = f'outputs/results/metrics_{domain}_stratified.json'
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    
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
    
    with open(metrics_file, 'w') as f:
        json.dump({'test': convert_numpy(test_metrics)}, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_file}")
    
    # Create visualizations
    figures_dir = f'outputs/figures/{domain}_stratified'
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
        title=f"Feature Importance - {domain.capitalize()} (Stratified)",
        filename=f'feature_importance_{domain}'
    )
    
    logger.info(f"✓ Saved figures to {figures_dir}/")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"REPROCESSING COMPLETE - {domain.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"\nKey Results:")
    logger.info(f"  Test AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test ECE:      {test_metrics['ece']:.4f}")
    
    # Calculate train/test hallucination rates
    train_hall_rate = (y_train == 1).sum() / len(y_train)
    test_hall_rate = (y_test == 1).sum() / len(y_test)
    logger.info(f"  Train/Test Hallucination Rate:")
    logger.info(f"    Train: {train_hall_rate:.1%}")
    logger.info(f"    Test:  {test_hall_rate:.1%}")
    logger.info(f"    Gap:   {abs(train_hall_rate - test_hall_rate):.1%}")
    
    logger.info(f"\nFiles:")
    logger.info(f"  Processed: {output_file}")
    logger.info(f"  Features:  {features_file}")
    logger.info(f"  Model:     {model_path}")
    logger.info(f"  Metrics:   {metrics_file}")
    logger.info(f"  Figures:   {figures_dir}/")


if __name__ == '__main__':
    main()
