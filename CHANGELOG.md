# Changelog

## 2026-02-07: Fix XGBoost Overfitting (Critical)

### Problem

The XGBoost classifier was catastrophically overfitting across all domains. The model memorized the training data but failed to generalize. This was most visible on the `is_agents` domain (smallest dataset):

| Metric | Train | Test |
|--------|-------|------|
| AUROC | 0.9997 | 0.5208 |
| Accuracy | 0.9875 | 0.4500 |
| F1 | 0.9851 | 0.3529 |

All other domains showed the same pattern (Train AUROC near 1.0 with large test gaps).

### Root Causes

1. **Extreme feature-to-sample ratio**: 41 features on as few as 80 training samples (is_agents). Safe minimum is ~10:1.
2. **Hyperparameter tuning never invoked**: `HyperparameterTuner` existed but `run_pipeline.py` called `classifier.train()` directly, bypassing it.
3. **No regularization**: `gamma=0`, `reg_alpha=0` meant no L1 regularization or minimum loss reduction.
4. **No early stopping**: All 200 estimators trained regardless of validation performance.
5. **Overly complex model for data size**: `max_depth=6`, `n_estimators=200`.
6. **Feature selection disabled**: Configured but never enabled.
7. **Tuner search space too narrow**: Optuna objective didn't tune `gamma`, `reg_alpha`, or `reg_lambda`.

### Changes Made

**configs/model.yaml** - Conservative defaults:
- `max_depth`: 6 → 3
- `n_estimators`: 200 → 100
- `min_child_weight`: 1 → 3
- `gamma`: 0 → 0.5
- `reg_alpha`: 0 → 0.5
- `reg_lambda`: 1 → 3
- Added `gamma`, `reg_alpha`, `reg_lambda` to Optuna search space
- Reduced `n_trials` to 20 for pipeline runs (increase to 100 for production)

**configs/features.yaml** - Enable feature selection:
- `feature_selection.enabled`: false → true

**src/classifier/xgboost_model.py** - Feature selection + early stopping:
- Added `select_features()` method using mutual information scoring
- Dynamic k_best cap: `min(k_best, n_samples // 5)` enforces healthy sample-to-feature ratio
- Added NaN handling before feature selection (median imputation)
- Internal 80/20 validation split for early stopping when no val set provided
- Persist selected features through save/load for consistent prediction
- Set `n_jobs=1` to avoid multiprocessing crashes

**src/classifier/hyperparameter_tuning.py** - Expanded search space:
- Added `gamma` [0, 5], `reg_alpha` [0, 5], `reg_lambda` [1, 10] to Optuna objective
- Adaptive CV folds: `min(cv_folds, len(y) // 10)` for small datasets
- Set `n_jobs=1` in both XGBoost and cross_val_score

**run_pipeline.py** - Pipeline improvements:
- Integrated `HyperparameterTuner` into Step 5 (tune before train)
- Skip steps 1-4 when `--skip-features` is set (avoids loading heavy ML models)

**src/evaluation/visualization.py** - Matplotlib fix:
- Force `Agg` backend to prevent crashes on headless/constrained environments

### Results (is_agents domain)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Train AUROC | 0.9997 | 0.6738 | Healthy reduction |
| Test AUROC | 0.5208 | **0.6250** | +0.10 |
| Test Accuracy | 0.4500 | **0.6000** | +0.15 |
| Test F1 | 0.3529 | **0.5556** | +0.20 |
| Test ECE | 0.3118 | **0.1283** | -0.18 (better calibrated) |
| Train/Test gap | 0.479 | **0.059** | Overfitting eliminated |

### Action Required

**All domains must be re-run** with `--skip-features` to apply the fix to training and evaluation. The feature CSVs are still valid; only the model training needs to be redone:

```bash
for domain in math medicine finance psychology is_agents; do
    python run_pipeline.py --domain $domain --skip-inference --skip-features
done
```
