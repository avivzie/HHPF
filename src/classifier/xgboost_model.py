"""
XGBoost classifier for hallucination prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif, VarianceThreshold
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import pickle

from src.utils import load_config, ensure_dir
from src.features.feature_aggregator import FeatureAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mutual_info_classif_with_seed(X, y):
    """Wrapper for mutual_info_classif with fixed random_state for reproducibility."""
    return mutual_info_classif(X, y, random_state=42)


class HallucinationClassifier:
    """XGBoost classifier for hallucination prediction."""
    
    def __init__(self, config_name: str = "model"):
        """
        Initialize classifier.
        
        Args:
            config_name: Configuration file name
        """
        self.config = load_config(config_name)['xgboost']
        self.feature_config = load_config("features").get('feature_selection', {})
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.selected_features = None
        self.feature_selector = None
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top-k features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Labels
            
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        method = self.feature_config.get('method', 'mutual_info')
        k_best = self.feature_config.get('k_best', 15)
        
        # Step 1: Remove zero-variance (constant) features first
        variance_selector = VarianceThreshold(threshold=0.0)
        X_var_filtered = variance_selector.fit_transform(X)
        var_feature_mask = variance_selector.get_support()
        var_filtered_features = X.columns[var_feature_mask].tolist()
        
        n_constant = len(X.columns) - len(var_filtered_features)
        if n_constant > 0:
            logger.info(f"Removed {n_constant} constant features (zero variance)")
        
        X = pd.DataFrame(X_var_filtered, columns=var_filtered_features, index=X.index)
        
        # Step 2: Dynamically cap k_best based on sample size (enforce 5:1 sample-to-feature ratio)
        max_features = max(1, len(y) // 5)
        k_best = min(k_best, max_features, len(X.columns))
        
        logger.info(f"Feature selection: method={method}, k_best={k_best} (max={max_features} for {len(y)} samples)")
        
        # Step 3: Select scoring function
        if method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'chi2':
            score_func = chi2
            # chi2 requires non-negative features
            if (X < 0).any().any():
                logger.warning("chi2 requires non-negative features, using mutual_info instead")
                score_func = mutual_info_classif
        elif method == 'f_classif':
            score_func = f_classif
        else:
            logger.warning(f"Unknown method '{method}', using mutual_info")
            score_func = mutual_info_classif
        
        # Step 4: Perform feature selection on non-constant features
        # Note: mutual_info_classif uses randomness, use wrapper with fixed seed for reproducibility
        if method == 'mutual_info':
            score_func = mutual_info_classif_with_seed
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k_best)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_mask = self.feature_selector.get_support()
        selected_features = [col for col, selected in zip(X.columns, feature_mask) if selected]
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        
        return X_selected_df, selected_features
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            features_df: Feature DataFrame
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Get feature columns
        aggregator = FeatureAggregator()
        feature_cols = aggregator.get_feature_columns(features_df)
        
        # Separate features and labels
        X = features_df[feature_cols]
        y = features_df['hallucination_label']
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Handle any remaining NaN values before feature selection
        if X.isnull().any().any():
            logger.warning(f"Found NaN values in features, filling with median")
            X = X.fillna(X.median())
            # Fill any remaining NaNs (if column is all NaN) with 0
            X = X.fillna(0)
        
        # Perform feature selection if enabled
        if self.feature_config.get('enabled', False):
            logger.info("Feature selection enabled")
            X, self.selected_features = self.select_features(X, y)
            logger.info(f"Selected {len(self.selected_features)} features from {len(feature_cols)}")
        
        # Split data
        if 'split' in features_df.columns:
            # Use predefined split
            X_train = X[features_df['split'] == 'train']
            X_test = X[features_df['split'] == 'test']
            y_train = y[features_df['split'] == 'train']
            y_test = y[features_df['split'] == 'test']
            
            logger.info(f"Using predefined train/test split")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            logger.info(f"Created random train/test split ({1-test_size:.0%}/{test_size:.0%})")
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Hallucination rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained model
        """
        # Create internal validation split if not provided (for early stopping)
        use_internal_val = False
        if X_val is None and y_val is None and len(y_train) > 20:
            # Split train into train/val (80/20) for early stopping
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )
            X_train = X_train_split
            y_train = y_train_split
            use_internal_val = True
            logger.info(f"Created internal validation set: {len(X_train)} train, {len(X_val)} val")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = self.config.get('scale_pos_weight', 'auto')
        if scale_pos_weight == 'auto':
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            
            # Apply multiplier if specified (for domains with severe imbalance)
            multiplier = self.config.get('scale_pos_weight_multiplier', 1.0)
            scale_pos_weight = scale_pos_weight * multiplier
            
            logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
            if multiplier != 1.0:
                logger.info(f"  Applied multiplier: {multiplier:.2f}x (increases sensitivity to hallucinations)")
        
        # Initialize model
        model_params = {
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'n_estimators': self.config.get('n_estimators', 200),
            'min_child_weight': self.config.get('min_child_weight', 1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'gamma': self.config.get('gamma', 0),
            'reg_alpha': self.config.get('reg_alpha', 0),
            'reg_lambda': self.config.get('reg_lambda', 1),
            'objective': self.config.get('objective', 'binary:logistic'),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': 1
        }
        
        self.model = xgb.XGBClassifier(**model_params)
        
        # Train with early stopping if validation set is available
        if X_val is not None and y_val is not None:
            early_stopping_rounds = self.config.get('early_stopping_rounds', 20)
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            logger.info(f"✓ Training complete with early stopping")
        else:
            self.model.fit(X_train, y_train, verbose=False)
            logger.info(f"✓ Training complete")
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels (0/1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Apply feature selection if it was used during training
        if self.selected_features is not None:
            X = X[self.selected_features]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Apply feature selection if it was used during training
        if self.selected_features is not None:
            X = X[self.selected_features]
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            set_name: Dataset name (for logging)
            
        Returns:
            Dictionary with metrics
        """
        # Predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auroc': roc_auc_score(y, y_proba),
        }
        
        # Log results
        logger.info(f"\n{set_name.upper()} SET RESULTS:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
        
        return metrics
    
    def evaluate_thresholds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Evaluate model at different decision thresholds.
        
        Args:
            X: Features
            y: True labels
            thresholds: List of thresholds to test (default: [0.3, 0.35, 0.4, 0.45, 0.5])
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
        
        # Get probabilities
        y_proba = self.predict_proba(X)
        
        results = []
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Log results
        logger.info("\n" + "="*70)
        logger.info("THRESHOLD ANALYSIS")
        logger.info("="*70)
        for _, row in results_df.iterrows():
            logger.info(f"\nThreshold: {row['threshold']:.2f}")
            logger.info(f"  Accuracy:   {row['accuracy']:.4f}")
            logger.info(f"  Precision:  {row['precision']:.4f}")
            logger.info(f"  Recall:     {row['recall']:.4f}")
            logger.info(f"  F1 Score:   {row['f1']:.4f}")
            logger.info(f"  Specificity: {row['specificity']:.4f}")
            logger.info(f"  Confusion: TP={row['true_positives']}, TN={row['true_negatives']}, FP={row['false_positives']}, FN={row['false_negatives']}")
        
        # Find best threshold by F1 score
        best_idx = results_df['f1'].idxmax()
        best_row = results_df.iloc[best_idx]
        logger.info("\n" + "="*70)
        logger.info(f"BEST THRESHOLD (by F1): {best_row['threshold']:.2f}")
        logger.info(f"  F1={best_row['f1']:.4f}, Precision={best_row['precision']:.4f}, Recall={best_row['recall']:.4f}")
        logger.info("="*70 + "\n")
        
        return results_df
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_k: Return only top k features
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None or self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        # Use selected features if feature selection was applied
        feature_names = self.selected_features if self.selected_features is not None else self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_k:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def save_model(self, output_path: str):
        """
        Save trained model.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        ensure_dir(Path(output_path).parent)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'selected_features': self.selected_features,
            'feature_selector': self.feature_selector
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {output_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance')
        self.selected_features = model_data.get('selected_features')
        self.feature_selector = model_data.get('feature_selector')
        
        logger.info(f"✓ Model loaded from {model_path}")


def main():
    """Main entry point for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost hallucination classifier")
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature matrix CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/models/xgboost_model.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Initialize classifier
    classifier = HallucinationClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        features_df,
        test_size=args.test_size
    )
    
    # Train model
    logger.info("\nTraining XGBoost classifier...")
    classifier.train(X_train, y_train)
    
    # Evaluate on training set
    train_metrics = classifier.evaluate(X_train, y_train, set_name="train")
    
    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test, set_name="test")
    
    # Feature importance
    importance_df = classifier.get_feature_importance(top_k=15)
    logger.info("\nTop 15 Most Important Features:")
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    classifier.save_model(args.output)
    
    # Save importance
    importance_path = Path(args.output).parent / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to {importance_path}")
    
    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()
