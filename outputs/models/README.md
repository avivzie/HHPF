# Final Thesis Models

**Training Date:** February 14, 2026  
**Status:** ✅ Thesis-ready and used in all results

## The 5 Final Models

These are the exact models used to generate all thesis figures and results:

| Domain | Model File | Samples | AUROC | Accuracy | ECE | Hallucination Rate |
|--------|------------|---------|-------|----------|-----|-------------------|
| **Math** | `xgboost_math.pkl` | 542 | **0.7973** | 0.806 | 0.184 | 29.0% |
| **IS Agents** | `xgboost_is_agents.pkl` | 500 | **0.7027** | 0.810 | 0.341 | 87.8% |
| **Psychology** | `xgboost_psychology.pkl` | 500 | **0.6715** | 0.740 | 0.344 | 25.0% |
| **Finance** | `xgboost_finance.pkl` | 500 | **0.6320** | 0.640 | 0.196 | 73.8% |
| **Medicine** | `xgboost_medicine.pkl` | 500 | **0.6192** | 0.690 | 0.057 | 49.6% |

**Mean AUROC:** 0.6845 ± 0.0711

## Training Configuration

All models trained with identical XGBoost configuration:
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss'
)
```

## Features Used

All models trained on **Full Feature Set** (41-48 features depending on domain):
- **Semantic Features** (3): semantic_entropy, semantic_energy, num_semantic_clusters
- **Contextual Features** (21-24): Entity, question type, token counts, lexical features
- **Naive Baselines** (4): mean_logprob, min_logprob, max_prob, perplexity
- **Domain-Specific** (varies): MCQ features for applicable domains

## Reproducibility

To reproduce these exact models:

```bash
# Use the cached features (no API calls needed)
python scripts/train_consistent_models.py
```

This will:
1. Load cached features from `data/features/{domain}_features.csv`
2. Use predefined train/test splits (80/20, stratified)
3. Train with `random_state=42` for reproducibility
4. Save models to this directory

## Verification

To verify consistency across all outputs:

```bash
python scripts/verify_consistency.py
```

This checks that:
- Individual domain ROC curves match RQ3b AUROC values
- Ablation study results are consistent
- All metrics files match model outputs

## Used In

These models are used to generate:
- ✅ Individual domain ROC curves (`outputs/figures/{domain}/roc_curve_{domain}.pdf`)
- ✅ RQ3b domain AUROC comparison (`outputs/research_questions/figures/rq3b_domain_auroc.pdf`)
- ✅ All domain-specific metrics (`outputs/results/metrics_{domain}.json`)
- ✅ All confusion matrices, calibration plots, ARC curves

## Archive

Old experimental models (calibration tests) are in:
- `archive_old_experiments/` - Not used in thesis, kept for reference

## Important

**These 5 files are the single source of truth for all thesis results.**

Do NOT delete or modify these models unless regenerating all results from scratch.
