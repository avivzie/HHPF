# Outputs

This directory contains all research outputs: trained models, figures, and results.

## Directory Structure

```
outputs/
├── models/          # Trained XGBoost models
├── figures/         # Publication-ready plots
└── results/         # Metrics and analysis tables
```

## Models

Trained classifiers saved as:
- `xgboost_{domain}.pkl` - Domain-specific models
- `xgboost_unified.pkl` - Model trained on all domains
- `model_metadata.json` - Hyperparameters and training info

## Figures

Publication-quality plots for thesis:

1. **roc_curves.pdf** - ROC curves (full model vs baselines)
2. **arc_comparison.pdf** - Accuracy-Rejection Curves by domain
3. **ece_calibration.pdf** - Expected Calibration Error diagrams
4. **feature_importance_global.pdf** - Global feature importance
5. **feature_importance_by_domain.pdf** - Domain-specific importance
6. **correlation_heatmap.pdf** - Feature correlation matrix
7. **domain_comparison.pdf** - Hallucination rates by domain
8. **knowledge_overshadowing.pdf** - Entity rarity vs hallucination

All figures saved as:
- **PDF** (vector graphics for LaTeX)
- **PNG** (high-resolution for presentations)

## Results

Quantitative results saved as CSV:

- `metrics_summary.csv` - AUROC, Accuracy, Precision, Recall
- `feature_importance.csv` - Feature rankings with SHAP values
- `ablation_study.csv` - Baseline comparisons
- `domain_analysis.csv` - Per-domain statistics
- `statistical_tests.csv` - Hypothesis test results (p-values)

## Reproducibility

All outputs include metadata:
- Timestamp
- Configuration used
- Random seed
- Model version
- Dataset version

To regenerate all outputs:
```bash
python -m src.evaluation.evaluate_model --regenerate-all
```
