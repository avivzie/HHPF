"""
Train consistent models across all domains for thesis-ready results.

This script ensures COMPLETE CONSISTENCY by:
1. Using identical XGBoost configuration across all domains
2. Training all 5 feature subsets (Naive, Semantic, Context, Semantic+Context, Full)
3. Using predefined train/test splits from cached features
4. Calculating comprehensive metrics (AUROC, ARC, ECE, ROC curves, etc.)
5. Saving models and metrics in standardized locations

Output:
- outputs/models/xgboost_{domain}.pkl (Full model)
- outputs/ablation/{domain}_ablation_results.csv (all 5 subsets)
- outputs/ablation/{domain}_feature_importance.csv (feature rankings)
- outputs/results/metrics_{domain}.json (comprehensive metrics for Full model)

Usage:
    python scripts/train_consistent_models.py --domain math
    python scripts/train_consistent_models.py --domain is_agents
    ...or all at once:
    for domain in math is_agents psychology medicine finance; do
        python scripts/train_consistent_models.py --domain $domain
    done
"""

import argparse
import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold

# Import HHPF modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def define_feature_subsets(all_features):
    """
    Define 5 feature subsets for ablation study.
    MUST match scripts/per_domain_ablation.py exactly.
    """
    # Naive-Only: Baseline confidence features
    naive_features = [f for f in all_features if any(
        kw in f for kw in ['naive_max_prob', 'naive_perplexity', 'mean_logprob', 'min_logprob']
    )]
    
    # Semantic-Only: Semantic uncertainty features
    semantic_features = [f for f in all_features if any(
        kw in f for kw in ['semantic_entropy', 'semantic_energy', 'num_semantic_clusters']
    )]
    
    # Context-Only: Contextual features
    contextual_features = [f for f in all_features if any(
        kw in f for kw in ['entity', 'rarity', 'token', 'lexical', 'parse', 'qtype', 'complexity']
    )]
    
    # Semantic+Context: Hybrid features (no naive)
    semantic_context = semantic_features + contextual_features
    
    # Full: All features
    full_features = all_features
    
    return {
        'Naive-Only': naive_features,
        'Semantic-Only': semantic_features,
        'Context-Only': contextual_features,
        'Semantic+Context': semantic_context,
        'Full': full_features
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train XGBoost model and evaluate on test set.
    MUST match scripts/per_domain_ablation.py exactly.
    """
    # Calculate class imbalance ratio
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Fixed XGBoost configuration (IDENTICAL to per_domain_ablation.py)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': random_state,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate basic metrics
    metrics = {
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics, model, y_pred, y_pred_proba


def cross_validate_model(X, y, random_state=42, n_folds=5):
    """
    Perform 5-fold stratified cross-validation.
    Uses same XGBoost config as train_and_evaluate(); scale_pos_weight computed per fold.

    Args:
        X: Full feature matrix (float)
        y: Full labels (0/1)
        random_state: Random seed
        n_folds: Number of folds (default 5)

    Returns:
        dict with *_cv_mean and *_cv_std for auroc, accuracy, precision, recall, f1
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aurocs, accs, precs, recs, f1s = [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_state,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',
            'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        aurocs.append(roc_auc_score(y_te, y_proba))
        accs.append(accuracy_score(y_te, y_pred))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recs.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    def mean_std(vals):
        arr = np.array(vals)
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(vals) > 1 else (float(np.mean(arr)), 0.0)

    auroc_mean, auroc_std = mean_std(aurocs)
    acc_mean, acc_std = mean_std(accs)
    prec_mean, prec_std = mean_std(precs)
    rec_mean, rec_std = mean_std(recs)
    f1_mean, f1_std = mean_std(f1s)

    return {
        'auroc_cv_mean': auroc_mean,
        'auroc_cv_std': auroc_std,
        'accuracy_cv_mean': acc_mean,
        'accuracy_cv_std': acc_std,
        'precision_cv_mean': prec_mean,
        'precision_cv_std': prec_std,
        'recall_cv_mean': rec_mean,
        'recall_cv_std': rec_std,
        'f1_cv_mean': f1_mean,
        'f1_cv_std': f1_std,
    }


def convert_numpy_to_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    return obj


def train_domain(domain: str, features_path: str, output_dir: str, models_dir: str, metrics_dir: str):
    """
    Train all models for one domain with complete consistency.
    """
    logger.info("="*80)
    logger.info(f"TRAINING CONSISTENT MODELS: {domain.upper()}")
    logger.info("="*80)
    
    # Load features
    logger.info(f"\nLoading features from {features_path}")
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Identify feature columns (exclude metadata and text columns)
    exclude_cols = ['prompt_id', 'domain', 'split', 'hallucination_label', 
                    'response_id', 'question', 'answer', 'source',
                    'ground_truth', 'primary_response', 'context', 'documents']
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    # Also exclude any columns with object dtype (text columns)
    numeric_features = []
    for col in all_features:
        if df[col].dtype in ['int64', 'float64', 'bool']:
            numeric_features.append(col)
    
    all_features = numeric_features
    logger.info(f"Total numeric features: {len(all_features)}")
    
    # Check for split column
    if 'split' not in df.columns:
        logger.error("No 'split' column found - cannot proceed")
        raise ValueError("Missing 'split' column in features")
    
    # Split data
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    logger.info(f"\nTrain set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Train hallucination rate: {train_df['hallucination_label'].mean():.1%}")
    logger.info(f"Test hallucination rate: {test_df['hallucination_label'].mean():.1%}")
    
    # Define feature subsets
    feature_subsets = define_feature_subsets(all_features)
    
    logger.info(f"\nFeature subsets:")
    for name, features in feature_subsets.items():
        logger.info(f"  {name}: {len(features)} features")

    # Full dataset for cross-validation (train + test combined)
    full_df = df
    y_full = full_df['hallucination_label'].values

    # Train all feature subsets
    ablation_results = []
    full_model = None
    full_model_features = None

    for subset_name, subset_features in feature_subsets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {subset_name}")
        logger.info(f"{'='*60}")
        
        if len(subset_features) == 0:
            logger.warning(f"No features found for {subset_name} - skipping")
            continue
        
        # Filter features that exist in the dataframe
        available_features = [f for f in subset_features if f in df.columns]
        if len(available_features) < len(subset_features):
            logger.warning(f"Only {len(available_features)}/{len(subset_features)} features available")
        
        if len(available_features) == 0:
            logger.warning(f"No available features for {subset_name} - skipping")
            continue
        
        # Prepare data
        X_train = train_df[available_features].values
        y_train = train_df['hallucination_label'].values
        X_test = test_df[available_features].values
        y_test = test_df['hallucination_label'].values
        X_full = full_df[available_features].values

        # Train and evaluate (single split)
        try:
            metrics, model, y_pred, y_pred_proba = train_and_evaluate(X_train, y_train, X_test, y_test)

            # 5-fold cross-validation on full dataset
            cv_results = cross_validate_model(X_full, y_full, random_state=42, n_folds=5)

            logger.info(f"Results (single split):")
            logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"Results (5-fold CV):   AUROC {cv_results['auroc_cv_mean']:.4f} ± {cv_results['auroc_cv_std']:.4f}")

            # Store results (single-split + CV)
            row = {
                'feature_subset': subset_name,
                'n_features': len(available_features),
                'auroc': metrics['auroc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            }
            row.update(cv_results)
            ablation_results.append(row)
            
            # Save Full model for later use
            if subset_name == 'Full':
                full_model = model
                full_model_features = available_features
                full_y_test = y_test
                full_y_pred = y_pred
                full_y_pred_proba = y_pred_proba
                
                # Save feature importance
                importance_df = pd.DataFrame({
                    'feature': available_features,
                    'importance': model.feature_importances_
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df['rank'] = range(1, len(importance_df) + 1)
                
                importance_path = Path(output_dir) / f"{domain}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"\nSaved feature importance to {importance_path}")
                logger.info(f"Top 5 features:")
                for _, row in importance_df.head(5).iterrows():
                    logger.info(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")
        
        except Exception as e:
            logger.error(f"Error training {subset_name}: {e}")
            continue
    
    # Save ablation results
    if ablation_results:
        results_df = pd.DataFrame(ablation_results)
        ablation_path = Path(output_dir) / f"{domain}_ablation_results.csv"
        results_df.to_csv(ablation_path, index=False)
        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION RESULTS SAVED")
        logger.info(f"{'='*80}")
        logger.info(f"\nSaved to {ablation_path}")
        logger.info(f"\nSummary:")
        logger.info(results_df[['feature_subset', 'n_features', 'auroc']].to_string(index=False))
    else:
        logger.error("No ablation results generated")
        raise ValueError("No ablation results")
    
    # Save Full model
    if full_model is not None:
        model_path = Path(models_dir) / f"xgboost_{domain}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"\n✓ Saved Full model to {model_path}")
        
        # Calculate comprehensive metrics for Full model
        logger.info(f"\nCalculating comprehensive metrics for Full model...")
        calculator = MetricsCalculator()
        comprehensive_metrics = calculator.calculate_all_metrics(
            full_y_test, full_y_pred, full_y_pred_proba
        )
        
        # Save comprehensive metrics JSON
        metrics_json_path = Path(metrics_dir) / f"metrics_{domain}.json"
        with open(metrics_json_path, 'w') as f:
            json.dump({
                'test': convert_numpy_to_json(comprehensive_metrics)
            }, f, indent=2)
        logger.info(f"✓ Saved comprehensive metrics to {metrics_json_path}")
        logger.info(f"  - AUROC: {comprehensive_metrics['auroc']:.4f}")
        logger.info(f"  - ECE: {comprehensive_metrics['ece']:.4f}")
        logger.info(f"  - AUC-ARC: {comprehensive_metrics['arc']['auc_arc']:.4f}")
    else:
        logger.error("Full model not trained - cannot save model or metrics")
        raise ValueError("Full model not trained")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ {domain.upper()} TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    
    return results_df, comprehensive_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train consistent models for thesis-ready results"
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['math', 'is_agents', 'psychology', 'medicine', 'finance'],
        help='Domain to train models for'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/features',
        help='Directory containing feature CSVs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/ablation',
        help='Output directory for ablation results'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='outputs/models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        default='outputs/results',
        help='Output directory for comprehensive metrics'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    
    # Construct features path
    features_path = Path(args.features_dir) / f"{args.domain}_features.csv"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return 1
    
    # Train models
    try:
        train_domain(
            domain=args.domain,
            features_path=str(features_path),
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir
        )
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
