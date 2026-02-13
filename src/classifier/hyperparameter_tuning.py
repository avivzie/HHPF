"""
Hyperparameter tuning for XGBoost classifier using Optuna.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Optional
import logging
import optuna
from optuna.samplers import TPESampler

from src.utils import load_config
from src.classifier.xgboost_model import HallucinationClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning for hallucination classifier."""
    
    def __init__(self):
        """Initialize tuner."""
        self.config = load_config("model")['xgboost']
        self.tuning_config = self.config.get('hyperparameter_tuning', {})
        self.best_params = None
        self.best_score = None
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial
            X: Training features
            y: Training labels
            
        Returns:
            Cross-validation AUROC score
        """
        # Define search space
        search_space = self.tuning_config.get('search_space', {})
        
        params = {
            'max_depth': trial.suggest_int(
                'max_depth',
                min(search_space.get('max_depth', [4, 10])),
                max(search_space.get('max_depth', [4, 10]))
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                min(search_space.get('learning_rate', [0.01, 0.1])),
                max(search_space.get('learning_rate', [0.01, 0.1])),
                log=True
            ),
            'n_estimators': trial.suggest_int(
                'n_estimators',
                min(search_space.get('n_estimators', [100, 300])),
                max(search_space.get('n_estimators', [100, 300]))
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                min(search_space.get('min_child_weight', [1, 5])),
                max(search_space.get('min_child_weight', [1, 5]))
            ),
            'subsample': trial.suggest_float(
                'subsample',
                min(search_space.get('subsample', [0.7, 0.9])),
                max(search_space.get('subsample', [0.7, 0.9]))
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                min(search_space.get('colsample_bytree', [0.7, 0.9])),
                max(search_space.get('colsample_bytree', [0.7, 0.9]))
            ),
            'gamma': trial.suggest_float(
                'gamma',
                min(search_space.get('gamma', [0, 5])),
                max(search_space.get('gamma', [0, 5]))
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                min(search_space.get('reg_alpha', [0, 5])),
                max(search_space.get('reg_alpha', [0, 5]))
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                min(search_space.get('reg_lambda', [1, 10])),
                max(search_space.get('reg_lambda', [1, 10]))
            ),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': 1
        }
        
        # Calculate scale_pos_weight
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation (adapt folds for small datasets)
        cv_folds = self.tuning_config.get('cv_folds', 5)
        # Use fewer folds for small datasets (min 10 samples per fold)
        cv_folds = min(cv_folds, len(y) // 10)
        cv_folds = max(2, cv_folds)  # At least 2 folds
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1  # Single-threaded to avoid joblib/loky PermissionError in constrained environments
        )
        
        return scores.mean()
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: Optional[int] = None
    ) -> Dict:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Training features
            y: Training labels
            n_trials: Number of trials (default from config)
            
        Returns:
            Best parameters
        """
        if not self.tuning_config.get('enabled', False):
            logger.info("Hyperparameter tuning disabled, using default parameters")
            return self.config
        
        n_trials = n_trials or self.tuning_config.get('n_trials', 100)
        
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"\n✓ Hyperparameter tuning complete!")
        logger.info(f"  Best AUROC: {self.best_score:.4f}")
        logger.info(f"  Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"    {param}: {value}")
        
        return self.best_params
    
    def get_tuned_config(self) -> Dict:
        """
        Get configuration with tuned parameters.
        
        Returns:
            Updated configuration dictionary
        """
        if self.best_params is None:
            logger.warning("No tuned parameters available, returning default config")
            return self.config
        
        # Merge tuned params with config
        config = self.config.copy()
        config.update(self.best_params)
        
        return config


def main():
    """Main entry point for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature matrix CSV'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna trials'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/models/tuned_params.json',
        help='Output path for best parameters'
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Prepare data
    classifier = HallucinationClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df)
    
    # Tune hyperparameters
    tuner = HyperparameterTuner()
    best_params = tuner.tune(X_train, y_train, n_trials=args.n_trials)
    
    # Save best parameters
    import json
    from pathlib import Path
    from src.utils import ensure_dir
    
    ensure_dir(Path(args.output).parent)
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"✓ Best parameters saved to {args.output}")
    
    # Train final model with best parameters
    logger.info("\nTraining final model with tuned parameters...")
    config = tuner.get_tuned_config()
    
    # Update classifier config
    classifier.config.update(best_params)
    classifier.train(X_train, y_train)
    
    # Evaluate
    train_metrics = classifier.evaluate(X_train, y_train, set_name="train")
    test_metrics = classifier.evaluate(X_test, y_test, set_name="test")
    
    # Save tuned model
    output_model_path = str(Path(args.output).parent / "xgboost_tuned.pkl")
    classifier.save_model(output_model_path)


if __name__ == "__main__":
    main()
