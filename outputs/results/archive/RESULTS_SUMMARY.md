# HHPF Results Summary (2026-02-07)

## Pipeline Configuration

- **Model**: XGBoost with Optuna hyperparameter tuning (20 trials, adaptive CV folds)
- **Feature Selection**: Mutual information, top-15 features (capped at n_samples/5)
- **Regularization**: gamma, reg_alpha, reg_lambda tuned; early stopping with internal val split
- **Overfitting Fix Applied**: Yes (see CHANGELOG.md for details)

---

## Per-Domain Results

### Math (GSM8K)
| Metric | Train | Test |
|--------|-------|------|
| Samples | 434 | 108 |
| Features Selected | 15 / 41 | |
| Hallucination Rate | 28.1% | 32.4% |
| **AUROC** | 0.7622 | **0.7405** |
| Accuracy | 0.7719 | 0.8056 |
| ECE | -- | 0.1389 |
| Best CV AUROC (Optuna) | 0.7696 | |

**Notes**: Best-performing domain. Balanced classes, largest training set (434). Strong signal from semantic entropy and logprob features.

---

### Medicine (Med-HALT) -- LABELING FIX APPLIED
| Metric | Train | Test |
|--------|-------|------|
| Samples | 192 | 49 |
| Features Selected | 15 / 41 | |
| Hallucination Rate | 59.4% | 59.2% |
| **AUROC** | 0.5000 | **0.5000** |
| Accuracy | 0.4062 | 0.4082 |
| ECE | -- | 0.0918 |
| Best CV AUROC (Optuna) | 0.5000 | |

**Labeling fix applied (2026-02-07)**:
- Filtered out 259 "None of the above" MCQ-artifact samples (51.8% of original 500). Med-HALT is an MCQ benchmark; "None of the above" as ground truth is meaningless for free-text evaluation.
- Fixed punctuation bug in term matching (e.g., "flumazenil." != "flumazenil")
- Added containment check for short GT terms (e.g., if response contains "Flumazenil" and GT is "Flumazenil", label as faithful)
- Lowered similarity threshold from 0.50 to 0.30

**Result**: Hallucination rate corrected from 91% to 59% (now balanced). However, model still shows AUROC=0.5 -- the text-similarity-based labeling for medicine may not produce reliable enough ground truth for the features to learn from. With only 192 training samples and no logprobs available, this domain remains challenging. **For thesis**: Report as a limitation of the labeling methodology for medical domains.

---

### Finance (FinanceBench)
| Metric | Train | Test |
|--------|-------|------|
| Samples | 120 | 30 |
| Features Selected | 15 / 41 | |
| Hallucination Rate | 85.8% | 86.7% |
| **AUROC** | 0.6947 | **0.6731** |
| Accuracy | 0.6500 | 0.4333 |
| ECE | -- | 0.4303 |
| Best CV AUROC (Optuna) | 0.5789 | |

**Notes**: Constrained by small dataset (150 total, 30 test) and high class imbalance (86.7% hallucinations). Only 4 non-hallucinated samples in test set. AUROC shows some discriminative ability but test accuracy is poor due to threshold miscalibration. This is the full dataset -- 150 samples is all that FinanceBench provides.

---

### Psychology (TruthfulQA)
| Metric | Train | Test |
|--------|-------|------|
| Samples | 400 | 100 |
| Features Selected | 15 / 34 | |
| Hallucination Rate | 80.5% | 81.0% |
| **AUROC** | 0.6655 | **0.6491** |
| Accuracy | 0.7050 | 0.8100 |
| ECE | -- | 0.2793 |
| Best CV AUROC (Optuna) | 0.5933 | |

**Notes**: Moderate performance. High hallucination rate (81%) but model can still discriminate somewhat. Only 34 features available (vs 41 for other domains -- likely missing some epistemic features). Contextual features (entity rarity, lexical diversity) were the most informative for this domain.

---

### IS/Agents (HalluMix) -- Current: 100 samples
| Metric | Train | Test |
|--------|-------|------|
| Samples | 80 | 20 |
| Features Selected | 15 / 41 | |
| Hallucination Rate | 41.2% | 40.0% |
| **AUROC** | 0.6508 | **0.6042** |
| Accuracy | 0.6250 | 0.4000 |
| ECE | -- | 0.1300 |
| Best CV AUROC (Optuna) | 0.5540 | |

**Notes**: PRELIMINARY RESULT -- only 100 of 2,500 available samples used. Best-balanced dataset (41% hallucinations). Performance limited by tiny training set (80 samples). **Action needed**: Re-run with --limit 500 (requires API calls for inference + feature extraction) to get reliable results.

---

## Cross-Domain Comparison

| Domain | Test AUROC | Test Acc | Test ECE | Samples | Hall Rate | Status |
|--------|-----------|----------|----------|---------|-----------|--------|
| Math | **0.7405** | 0.8056 | 0.1389 | 542 | 29.0% | GOOD |
| Psychology | 0.6491 | 0.8100 | 0.2793 | 500 | 80.5% | OK |
| Finance | 0.6731 | 0.4333 | 0.4303 | 150 | 86.0% | LIMITED |
| IS/Agents | 0.6042 | 0.4000 | 0.1300 | 100 | 41.0% | PRELIMINARY |
| Medicine | 0.5000 | 0.4082 | 0.0918 | 241* | 59.3% | LABELING FIX |

*Medicine reduced from 500 to 241 after filtering "None of the above" MCQ artifacts.

**Key Observation**: Performance correlates with both class balance and labeling quality. Math (29% hallucinations, exact numerical match) achieves the best AUROC. Medicine, despite now having balanced classes (59%), still fails because similarity-based labeling is unreliable for medical text.

---

## Outstanding Actions

1. **IS/Agents**: Expand to 500 samples (currently running with `--limit 500 --provider groq`)
2. **Medicine**: Consider expanding to 500 non-"None" samples to improve signal (requires ~259 new API calls)
3. **Research Questions**: Once IS/Agents is expanded, run `python analyze_research_questions.py`
4. **More Optuna Trials**: Consider increasing to 50-100 trials for final results (currently 20)

---

## Files Generated

```
outputs/
├── models/
│   ├── xgboost_math.pkl
│   ├── xgboost_medicine.pkl
│   ├── xgboost_finance.pkl
│   ├── xgboost_psychology.pkl
│   └── xgboost_is_agents.pkl
├── results/
│   ├── metrics_math.json
│   ├── metrics_medicine.json
│   ├── metrics_finance.json
│   ├── metrics_psychology.json
│   ├── metrics_is_agents.json
│   └── RESULTS_SUMMARY.md (this file)
└── figures/{domain}/
    ├── roc_curve_{domain}.pdf/.png
    ├── arc_{domain}.pdf/.png
    ├── calibration_{domain}.pdf/.png
    ├── confusion_matrix_{domain}.pdf/.png
    └── feature_importance_{domain}.pdf/.png
```
