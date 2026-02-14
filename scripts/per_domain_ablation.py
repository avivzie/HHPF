"""
Per-domain ablation study for RQ1 and RQ2.

This script runs ablation study for a single domain across 5 feature subsets:
1. Naive-Only: MaxProb, Perplexity, Mean/Min LogProb
2. Semantic-Only: Semantic Entropy, Semantic Energy, Cluster Count
3. Context-Only: Entity Rarity, Token Stats, Lexical Complexity, Parse Depth
4. Semantic+Context: Semantic + Context features
5. Full: All features

Methodology:
- No cross-domain mixing (train/evaluate within domain)
- Identical XGBoost config for all feature subsets
- Fixed train/test splits from existing data
- Reproducible (random_state=42)
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def define_feature_subsets(all_features):
    """
    Define 5 feature subsets for ablation study.
    
    Args:
        all_features: List of all feature column names
        
    Returns:
        Dictionary mapping subset name to list of features
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
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate class imbalance ratio
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Fixed XGBoost configuration (same for all subsets)
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
    
    # Calculate metrics
    metrics = {
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics, model


def run_ablation_study(domain: str, features_path: str, output_dir: str):
    """
    Run ablation study for one domain.
    
    Args:
        domain: Domain name (math, is_agents, psychology, medicine, finance)
        features_path: Path to domain features CSV
        output_dir: Output directory for results
    """
    logger.info("="*80)
    logger.info(f"PER-DOMAIN ABLATION STUDY: {domain.upper()}")
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
        else:
            logger.debug(f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})")
    
    all_features = numeric_features
    logger.info(f"Total numeric features: {len(all_features)}")
    
    # Check for split column
    if 'split' not in df.columns:
        logger.error("No 'split' column found - cannot proceed with ablation")
        return
    
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
    
    # Run ablation for each subset
    results = []
    
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
        
        # Train and evaluate
        try:
            metrics, model = train_and_evaluate(X_train, y_train, X_test, y_test)
            
            logger.info(f"Results:")
            logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
            
            # Store results
            results.append({
                'feature_subset': subset_name,
                'n_features': len(available_features),
                'auroc': metrics['auroc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
            
            # Save feature importance for Full model
            if subset_name == 'Full':
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
    if results:
        results_df = pd.DataFrame(results)
        output_path = Path(output_dir) / f"{domain}_ablation_results.csv"
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION STUDY COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nResults saved to {output_path}")
        logger.info(f"\nSummary:")
        logger.info(results_df[['feature_subset', 'n_features', 'auroc']].to_string(index=False))
    else:
        logger.error("No results generated - ablation study failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run per-domain ablation study for HHPF research questions"
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['math', 'is_agents', 'psychology', 'medicine', 'finance'],
        help='Domain to run ablation study on'
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
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Construct features path
    features_path = Path(args.features_dir) / f"{args.domain}_features.csv"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return
    
    # Run ablation study
    try:
        run_ablation_study(
            domain=args.domain,
            features_path=str(features_path),
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Ablation study failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
