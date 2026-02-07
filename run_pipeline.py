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
    provider: str = "together",
    skip_inference: bool = False,
    skip_features: bool = False,
    skip_training: bool = False
):
    """
    Run the complete HHPF pipeline.
    
    Args:
        domain: Domain to process
        limit: Limit number of samples (for testing)
        provider: API provider ('together' or 'groq')
        skip_inference: Skip inference step (use cached)
        skip_features: Skip feature extraction (use cached)
        skip_training: Skip model training (use existing)
    """
    logger.info("="*60)
    logger.info("HHPF PIPELINE - End-to-End Execution")
    logger.info("="*60)
    logger.info(f"Domain: {domain}")
    logger.info(f"Sample limit: {limit if limit else 'None'}")
    logger.info(f"API Provider: {provider}")
    
    features_path = f"data/features/{domain}_features.csv"
    
    # When skip_features is set, features CSV already contains labels and splits,
    # so we can skip steps 1-4 entirely (avoids loading heavy ML models).
    if not skip_features:
        # 1. Data Preparation (NO TRAIN/TEST SPLIT YET)
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Data Preparation (without split)")
        logger.info("="*60)
        
        from src.data_preparation.process_datasets import process_dataset
        
        try:
            processed_df = process_dataset(domain, limit=limit)
            dataset_path = f"data/processed/{domain}_processed.csv"
            logger.info(f"✓ Dataset processed: {len(processed_df)} samples (no split yet)")
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
            
            generator = ResponseGenerator(provider=provider)
            
            responses_df = generator.generate_for_dataset(
                dataset_path=dataset_path,
                output_dir="data/features",
                limit=limit
            )
            
            logger.info(f"✓ Responses generated: {len(responses_df)} samples")
        else:
            logger.info("⊳ Skipping inference (using cached responses)")
        
        # 3. Label Responses & Create Stratified Split
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Label Responses & Stratified Split")
        logger.info("="*60)
        logger.info("📝 Labeling all responses before splitting (enables proper stratification)")
        
        from src.data_preparation.label_responses import label_all_responses, save_labeled_responses
        
        labeled_df = label_all_responses(
            processed_csv=dataset_path,
            responses_dir="data/features",
            domain=domain,
            train_ratio=0.8,
            random_seed=42
        )
        
        # Save labeled responses with stratified split
        responses_path = f"data/features/responses_{domain}_processed.csv"
        save_labeled_responses(labeled_df, responses_path)
        logger.info(f"✓ Labeled responses with stratified split: {len(labeled_df)} samples")
        
        # 4. Feature Extraction
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Feature Extraction")
        logger.info("="*60)
        
        from src.features.feature_aggregator import FeatureAggregator
        
        aggregator = FeatureAggregator()
        
        # Detect MCQ format for medicine domain (has correct_index column)
        format_type = "free_text"
        if domain == "medicine":
            import pandas as pd
            _processed_df = pd.read_csv(dataset_path)
            if 'correct_index' in _processed_df.columns:
                format_type = "mcq"
                logger.info("Detected MCQ format for medicine domain - using MCQ-specific features")
        
        features_df = aggregator.aggregate_features(
            responses_csv=responses_path,
            output_path=features_path,
            format_type=format_type
        )
        
        logger.info(f"✓ Features extracted: {features_df.shape[1]} features")
    else:
        logger.info("⊳ Skipping steps 1-4 (using cached features from %s)", features_path)
    
    # 5. Model Training
    if not skip_training:
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Model Training")
        logger.info("="*60)
        
        from src.classifier.xgboost_model import HallucinationClassifier
        from src.classifier.hyperparameter_tuning import HyperparameterTuner
        import pandas as pd
        
        features_df = pd.read_csv(features_path)
        
        classifier = HallucinationClassifier()
        X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
        
        # Hyperparameter tuning (if enabled in config)
        tuner = HyperparameterTuner()
        if tuner.tuning_config.get('enabled', False):
            logger.info("Hyperparameter tuning enabled - running Optuna...")
            best_params = tuner.tune(X_train, y_train)
            # Update classifier config with best params
            classifier.config.update(best_params)
            logger.info("Using tuned hyperparameters for training")
        else:
            logger.info("Hyperparameter tuning disabled - using default config")
        
        classifier.train(X_train, y_train)
        
        train_metrics = classifier.evaluate(X_train, y_train, set_name="train")
        test_metrics = classifier.evaluate(X_test, y_test, set_name="test")
        
        model_path = f"outputs/models/xgboost_{domain}.pkl"
        classifier.save_model(model_path)
        
        logger.info(f"✓ Model trained - Test AUROC: {test_metrics['auroc']:.4f}")
    else:
        model_path = f"outputs/models/xgboost_{domain}.pkl"
        logger.info("⊳ Skipping training (using existing model)")
    
    # 6. Evaluation & Visualization
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Evaluation & Visualization")
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
    
    # For psychology domain with low recall, analyze different thresholds
    if domain == 'psychology':
        logger.info("\n" + "="*60)
        logger.info("Psychology Domain: Analyzing Decision Thresholds")
        logger.info("="*60)
        threshold_results = classifier.evaluate_thresholds(X_test, y_test)
        
        # Save threshold analysis
        threshold_path = f"outputs/results/threshold_analysis_{domain}.csv"
        threshold_results.to_csv(threshold_path, index=False)
        logger.info(f"✓ Threshold analysis saved to {threshold_path}")
    
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
        '--provider',
        type=str,
        choices=['together', 'groq'],
        default='together',
        help='API provider: together (default) or groq'
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
            provider=args.provider,
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
