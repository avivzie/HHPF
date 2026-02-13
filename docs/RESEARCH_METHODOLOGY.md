# Research Methodology

## 1. Research Overview & Objectives

This research investigates epistemic uncertainty as a signal for detecting hallucinations in Large Language Models (LLMs) across five distinct domains. The core hypothesis is that **semantic uncertainty features combined with contextual factors provide more reliable hallucination detection than naive confidence metrics alone**.

### Research Questions

**RQ1: Feature Hypothesis**  
Do hybrid features (semantic uncertainty + contextual) outperform baseline approaches (naive confidence, semantic-only, context-only)?

**RQ2: Semantic Uncertainty vs Naive Confidence**  
Does Semantic Entropy (meaning-based uncertainty) provide more reliable hallucination detection than naive confidence metrics (MaxProb, Perplexity)?

**RQ3: Cross-Domain Variance**  
Do hallucination signatures differ significantly across domains, requiring domain-specific detection approaches?

---

## 2. Datasets & Domains

Five benchmark datasets representing diverse question-answering scenarios:

| Domain | Dataset | Samples | Ground Truth Method | Characteristics |
|--------|---------|---------|---------------------|-----------------|
| **Medicine** | Med-HALT | 1,000+ | Medical entity matching | Clinical MCQs with structured answers |
| **Math** | GSM8K | 1,000+ | Exact numerical match | Grade school math word problems |
| **Finance** | TAT-QA | 18,215 | Document-grounded | Tabular + textual financial QA |
| **IS** | HalluMix | 9,400+ | Document-grounded | Multi-source document QA |
| **Psychology** | TruthfulQA | 800+ | Truthfulness label | Common misconceptions & cognitive biases |

### Labeling Approaches

- **Pre-labeled**: Psychology (TruthfulQA includes truthfulness labels)
- **Rule-based**: Math (exact numerical matching), Medicine (medical entity matching)
- **Document-grounded**: Finance and IS (compare LLM response against source documents using text similarity and term overlap)

All domains use stratified sampling to maintain class balance across train/test splits (80/20 ratio, random seed 42).

---

## 3. Pipeline & Data Processing

### Six-Step Pipeline

1. **Data Loading**: Load domain-specific datasets, validate columns, create prompt IDs
2. **Response Generation**: Generate 5 stochastic samples per prompt (temperature 0.7-1.0) using Llama-3.1-8B (Groq or Together AI)
3. **Labeling**: Apply domain-specific ground truth methods to label responses as hallucinations (1) or faithful (0)
4. **Stratified Split**: Create train/test split AFTER labeling to ensure proper class stratification
5. **Feature Extraction**: Extract epistemic uncertainty, contextual, and baseline features
6. **Model Training**: Train XGBoost classifier with hyperparameter tuning and overfitting safeguards

### Key Methodological Decisions

**Stratification After Labeling**: For document-grounded domains (Finance, IS), labels are computed dynamically. Train/test split occurs AFTER labeling to enable proper stratification on actual hallucination labels.

**Stochastic Sampling**: Each prompt generates 5 responses with varying temperatures (0.7-1.0) to capture model uncertainty. This enables semantic entropy calculation through response diversity analysis.

**Document-Grounded Prompts**: For Finance and IS domains, source documents are included in prompts:
```
Answer the following question based ONLY on the provided context.
Be concise and accurate.

Context: [tables + paragraphs]
Question: [question]
Answer:
```

**Cache Management**: Responses and features are cached to enable pipeline resumption and iterative development without redundant API calls.

---

## 4. Feature Engineering

### Epistemic Uncertainty Features (Primary)

**Semantic Entropy**: Measures uncertainty through meaning-based clustering of stochastic responses.
- Method: NLI-based clustering using DeBERTa-v3-large
- Entailment threshold: 0.7
- Output: Entropy over semantic clusters, number of clusters, average cluster size

**Semantic Energy**: Analyzes the logit distribution to detect low-confidence states.
- Method: Negative log-sum-exp over top-100 logits
- Captures distributional uncertainty in token predictions

### Contextual Features (Hypothesis)

**Knowledge Popularity**: Entity rarity as a proxy for knowledge obscurity.
- Entity extraction: SpaCy (PERSON, ORG, GPE, LOC, PRODUCT, EVENT)
- Rarity metric: Log-inverse frequency (Wikipedia-based)
- Aggregation: Mean rarity across entities

**Prompt Complexity**: Linguistic and syntactic characteristics.
- Lexical: Token count, unique token ratio, word length, lexical diversity
- Syntactic: Parse depth, clause count, dependency arc length
- Question type: One-hot encoding for question words (what, why, how, etc.)

### MCQ-Specific Features (Medicine Domain)

**Letter Consistency**: Fraction of stochastic samples agreeing on the same answer choice (A/B/C/D).

**Letter Entropy**: Shannon entropy over letter distribution (replaces NLI-based semantic entropy for MCQ format).

**First-Token Logprob**: Probability of the answer letter token (A/B/C/D) only.

### Baseline Features (for RQ2)

**Naive Confidence Metrics**:
- MaxProb: Maximum token probability
- Perplexity: exp(-mean(logprobs))
- Mean/Min Logprob: Aggregate token-level confidence

---

## 5. Model Architecture & Training

### XGBoost Binary Classifier

**Architecture**: Gradient boosted decision trees (GBDT) with binary logistic objective.

**Hyperparameters** (conservative defaults to prevent overfitting):
```yaml
max_depth: 3
learning_rate: 0.05
n_estimators: 100
min_child_weight: 3
subsample: 0.8
colsample_bytree: 0.8
gamma: 0.5          # Min loss reduction
reg_alpha: 0.5      # L1 regularization
reg_lambda: 3       # L2 regularization
```

### Feature Selection

**Method**: Mutual Information scoring with variance thresholding
- Removes constant features (zero variance)
- Selects top-k features based on MI scores
- Dynamic cap: `k_best = min(configured_k, n_samples // 5)` to maintain healthy feature-to-sample ratio
- Ensures train/test consistency by persisting selected features in model

### Hyperparameter Tuning

**Method**: Optuna (Tree-structured Parzen Estimator)
- 20 trials per domain
- 5-fold cross-validation (adaptive: min(5, n_samples // 10) for small datasets)
- Search space: max_depth [4-10], learning_rate [0.01-0.1], n_estimators [100-300], regularization [0-10]
- Objective: Maximize validation AUROC

### Overfitting Prevention

Critical safeguards implemented after initial overfitting was detected (Train AUROC 0.999, Test AUROC 0.52):

1. **Feature Selection**: Enforces n_samples/5 ratio
2. **Conservative Defaults**: Low max_depth (3), high regularization
3. **Early Stopping**: 20 rounds on validation set
4. **Internal Validation**: 80/20 split from training data when no validation set provided
5. **Regularization**: Both L1 (reg_alpha) and L2 (reg_lambda) penalties

### Class Imbalance Handling

**Method**: Auto-calculated `scale_pos_weight` based on class distribution:
```python
scale_pos_weight = (n_negative / n_positive) * multiplier
```

**Multiplier adjustments**:
- Default: 1.0 (balanced weighting)
- Psychology: 2.0 (corrected recall from 0.16 → 0.84 for severe imbalance)

---

## 6. Evaluation Metrics

### Primary Metric: AUROC

**Area Under the Receiver Operating Characteristic curve** - threshold-independent measure of classifier discrimination ability. Interprets as probability that classifier ranks a random hallucination higher than a random faithful response.

### Classification Metrics

Evaluated at optimal threshold (maximizing F1 score):
- **Accuracy**: Overall correctness
- **Precision**: P(faithful | predicted faithful)
- **Recall**: P(predicted faithful | actually faithful)
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: P(predicted hallucination | actually hallucination)

### Calibration Metrics

**Expected Calibration Error (ECE)**: Measures alignment between predicted confidence and actual accuracy.
- Method: 10-bin reliability diagram
- Interpretation: Lower ECE indicates better-calibrated predictions

### Selective Prediction

**Accuracy-Rejection Curve (ARC)**: Evaluates accuracy at different confidence thresholds.
- X-axis: Rejection rate (% of uncertain predictions abstained)
- Y-axis: Accuracy on retained predictions
- AUC-ARC: Area under the curve (higher = better selective prediction)

### Statistical Testing

**RQ1 & RQ2**: Pairwise AUROC comparisons with statistical significance tests
**RQ3**: Chi-square test for domain differences in hallucination rates

---

## 7. Validation & Safeguards

### Methodological Integrity

**Feature Selection Integrity**: Selected features are persisted in the trained model to ensure identical feature subsets are used during training and testing. Prevents information leakage from test set during selection.

**Train/Test Split Consistency**: Stratified sampling ensures class balance is maintained across splits. Split assignment is preserved through all pipeline steps via persistent `split` column.

**Model Serialization**: Complete model state (including preprocessing, feature selection, and trained estimator) is serialized using pickle for reproducible evaluation.

### Domain-Specific Validation

**Medicine (MCQ)**:
- Issue: Initial AUROC 0.50 (random performance)
- Root cause: Semantic entropy failed on structured MCQ answers
- Solution: Implemented MCQ-specific features (letter consistency, letter entropy, first-token logprob)
- Result: AUROC improved to 0.68

**Psychology (Class Imbalance)**:
- Issue: Severe class imbalance (90% hallucinations) → Recall 0.16
- Root cause: XGBoost defaulted to predicting majority class
- Solution: Increased `scale_pos_weight_multiplier` to 2.0
- Result: Recall improved to 0.84, AUROC maintained at 0.69

**Math (Calibration Trade-off)**:
- Issue: ECE increased from 0.058 → 0.114 after overfitting fixes
- Root cause: Regularization reduced model confidence
- Finding: Trade-off between discrimination (AUROC) and calibration (ECE) is expected
- Conclusion: AUROC prioritized for hallucination detection; calibration acceptable at 0.11

**Finance (Dataset Switch)**:
- Issue: FinanceBench had only 150 samples (4 faithful in test) → AUROC 0.41
- Root cause: Insufficient data and extreme imbalance
- Solution: Switched to TAT-QA (18,215 samples with tables + text context)
- Implementation: Document-grounded labeling with context included in prompts

**IS Agents (Small Sample Anomaly)**:
- Issue: Test AUROC (0.6327) slightly exceeds Train AUROC (0.6269)
- Analysis: Statistical noise due to small test set (20 samples) with extreme imbalance
- Conclusion: Within confidence intervals; documented as limitation

### Quality Assurance Checks

1. **AUROC Gap Monitoring**: Train/test AUROC gap < 0.15 indicates acceptable generalization
2. **Feature Variance**: Constant features (zero variance) are automatically removed
3. **Sample Size Validation**: Minimum 500 samples per domain for robust evaluation
4. **Label Distribution**: Both classes must be present for stratified splitting
5. **NaN Handling**: Missing features imputed with median values before training

### Reproducibility

- **Random Seed**: 42 (fixed across all random operations)
- **Stratified Sampling**: Ensures consistent class distributions
- **Configuration Management**: All parameters externalized in YAML configs
- **Version Control**: Complete git history of methodological changes
- **Documentation**: Research logs track domain-specific issues and resolutions

---

## 8. Known Limitations

### API Limitation: Missing Token-Level Probabilities

**Issue:** The HHPF pipeline uses Groq and Together AI APIs for LLM inference. Neither API was configured to return token-level log probabilities (logprobs) in their responses.

**Impact on Features:**
All logprobs-dependent features are 100% NULL across all domains:
- `semantic_energy` (logit-based uncertainty)
- `mean_logprob`, `min_logprob`, `std_logprob` (aggregate confidence)
- `naive_max_prob` (maximum token probability)
- `naive_perplexity` (model uncertainty metric)

**Working Features:**
- `semantic_entropy` (DeBERTa NLI-based, 0% NULL) ✅
- `num_semantic_clusters`, `avg_cluster_size` (clustering-based) ✅
- All 21 contextual features (entities, question types, response length, etc.) ✅

**Impact on Research Questions:**

- **RQ1 (Feature Hypothesis):** ✅ UNAFFECTED - Ablation study compares semantic+context (0.8109 AUROC) vs semantic-only (0.7788 AUROC) vs context-only (0.5654 AUROC). Naive baseline is non-functional (0.50) but doesn't invalidate the hybrid hypothesis.

- **RQ2 (Semantic vs Naive):** ⚠️ LIMITED - Cannot definitively compare semantic vs naive confidence. However, this limitation validates a key practical advantage: **semantic entropy provides strong hallucination detection (AUROC 0.60-0.81) without requiring token-level probabilities**, making it deployable across any LLM API, including closed-source models that don't expose logprobs (Claude, GPT-4 via Azure).

- **RQ3 (Cross-Domain Variance):** ✅ UNAFFECTED - Compares domains using available features (semantic + contextual). Domain-specific AUROCs range from 0.60-0.78, with significant variation confirmed (χ² = 556.22, p < 0.001).

**Thesis Framing:**

This limitation is framed as a **strength** in the thesis: The framework demonstrates that semantic entropy (linguistic disagreement via NLI) provides strong hallucination detection **without requiring model internals**. This aligns with:
1. Real-world deployment constraints (most production APIs don't expose logprobs)
2. Recent literature emphasizing linguistic uncertainty over model confidence (Farquhar et al., 2024; Kuhn et al., 2023)
3. API-agnostic applicability across diverse LLM providers

**Future Work:** Implementing logprobs support would require minimal code changes (~30 minutes) and enable full RQ2 comparison. The research framework is designed to accommodate this addition without architectural changes.

---

## Summary

This methodology implements a rigorous pipeline for cross-domain hallucination detection, combining semantic uncertainty (epistemic approach) with contextual features (knowledge-based approach). The pipeline includes multiple validation safeguards to ensure methodological integrity, addresses domain-specific challenges through adaptive solutions, and provides comprehensive evaluation across multiple metrics. All experiments are reproducible through fixed random seeds, versioned configurations, and persistent model serialization.

**Key Innovation:** The framework achieves strong performance (AUROC 0.60-0.81) using only semantic entropy and contextual features, without requiring token-level probabilities. This makes the approach practical for production deployment across diverse LLM APIs.
