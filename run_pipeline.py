"""
End-to-end pipeline for HHPF.

Orchestrates the complete workflow from data preparation to evaluation.
"""

import argparse
import logging
from pathlib import Path
import sys

from src.utils import load_config, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    domain: str = "math",
    limit: int = None,
    skip_inference: bool = False,
    skip_features: bool = False,
    skip_training: bool = False
):
    """
    Run the complete HHPF pipeline.
    
    Args:
        domain: Domain to process
        limit: Limit number of samples (for testing)
        skip_inference: Skip inference step (use cached)
        skip_features: Skip feature extraction (use cached)
        skip_training: Skip model training (use existing)
    """
    logger.info("="*60)
    logger.info("HHPF PIPELINE - End-to-End Execution")
    logger.info("="*60)
    logger.info(f"Domain: {domain}")
    logger.info(f"Sample limit: {limit if limit else 'None'}")
    
    # 1. Data Preparation
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Data Preparation")
    logger.info("="*60)
    
    from src.data_preparation.process_datasets import process_dataset
    
    try:
        processed_df = process_dataset(domain)
        dataset_path = f"data/processed/{domain}_processed.csv"
        logger.info(f"✓ Dataset processed: {len(processed_df)} samples")
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error(f"Please place your dataset in data/raw/")
        sys.exit(1)
    
    # 2. Inference (Response Generation)
    if not skip_inference:
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Response Generation")
        logger.info("="*60)
        
        from src.inference.response_generator import ResponseGenerator
        
        generator = ResponseGenerator(provider="together")
        
        responses_df = generator.generate_for_dataset(
            dataset_path=dataset_path,
            output_dir="data/features",
            limit=limit
        )
        
        responses_path = f"data/features/responses_{domain}_processed.csv"
        logger.info(f"✓ Responses generated: {len(responses_df)} samples")
    else:
        responses_path = f"data/features/responses_{domain}_processed.csv"
        logger.info("⊳ Skipping inference (using cached responses)")
    
    # 3. Feature Extraction
    if not skip_features:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Feature Extraction")
        logger.info("="*60)
        
        from src.features.feature_aggregator import FeatureAggregator
        
        aggregator = FeatureAggregator()
        
        features_df = aggregator.aggregate_features(
            responses_csv=responses_path,
            output_path=f"data/features/{domain}_features.csv"
        )
        
        features_path = f"data/features/{domain}_features.csv"
        logger.info(f"✓ Features extracted: {features_df.shape[1]} features")
    else:
        features_path = f"data/features/{domain}_features.csv"
        logger.info("⊳ Skipping feature extraction (using cached features)")
    
    # 4. Model Training
    if not skip_training:
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Model Training")
        logger.info("="*60)
        
        from src.classifier.xgboost_model import HallucinationClassifier
        import pandas as pd
        
        features_df = pd.read_csv(features_path)
        
        classifier = HallucinationClassifier()
        X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
        
        classifier.train(X_train, y_train)
        
        train_metrics = classifier.evaluate(X_train, y_train, set_name="train")
        test_metrics = classifier.evaluate(X_test, y_test, set_name="test")
        
        model_path = f"outputs/models/xgboost_{domain}.pkl"
        classifier.save_model(model_path)
        
        logger.info(f"✓ Model trained - Test AUROC: {test_metrics['auroc']:.4f}")
    else:
        model_path = f"outputs/models/xgboost_{domain}.pkl"
        logger.info("⊳ Skipping training (using existing model)")
    
    # 5. Evaluation & Visualization
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Evaluation & Visualization")
    logger.info("="*60)
    
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.visualization import HHPFVisualizer
    from src.classifier.xgboost_model import HallucinationClassifier
    import pandas as pd
    import json
    
    # Load model and features
    classifier = HallucinationClassifier()
    classifier.load_model(model_path)
    
    features_df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
    
    # Predictions
    y_pred_test = classifier.predict(X_test)
    y_proba_test = classifier.predict_proba(X_test)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    test_metrics = calculator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)
    
    # Save metrics
    metrics_path = f"outputs/results/metrics_{domain}.json"
    ensure_dir(Path(metrics_path).parent)
    
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    with open(metrics_path, 'w') as f:
        json.dump({'test': convert_numpy(test_metrics)}, f, indent=2)
    
    # Generate visualizations
    viz = HHPFVisualizer(output_dir=f"outputs/figures/{domain}")
    
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
    
    logger.info(f"✓ Evaluation complete - Results saved to outputs/")
    
    # 6. Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"\nKey Results for {domain.capitalize()}:")
    logger.info(f"  Test AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test ECE:      {test_metrics['ece']:.4f}")
    logger.info(f"\nOutputs:")
    logger.info(f"  Model:       {model_path}")
    logger.info(f"  Metrics:     {metrics_path}")
    logger.info(f"  Figures:     outputs/figures/{domain}/")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review figures in outputs/figures/{domain}/")
    logger.info(f"  2. Run hypothesis tests: python -m src.evaluation.hypothesis_testing --features {features_path}")
    logger.info(f"  3. Expand to other domains")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run HHPF pipeline end-to-end")
    parser.add_argument(
        '--domain',
        type=str,
        default='math',
        choices=['medicine', 'math', 'finance', 'is_agents', 'psychology'],
        help='Domain to process (default: math)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples (for testing)'
    )
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip inference step (use cached responses)'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature extraction (use cached features)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing model)'
    )
    
    args = parser.parse_args()
    
    try:
        run_full_pipeline(
            domain=args.domain,
            limit=args.limit,
            skip_inference=args.skip_inference,
            skip_features=args.skip_features,
            skip_training=args.skip_training
        )
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
