# Research Questions - Implementation Guide

## Overview

The HHPF system now fully supports answering all three research questions:

### RQ1: Feature Hypothesis
**Question:** Do hybrid features (semantic uncertainty + contextual) outperform baseline approaches?

### RQ2: Semantic Uncertainty vs Naive Confidence  
**Question:** Does Semantic Entropy (meaning-based) provide more reliable hallucination detection than naive confidence metrics (MaxProb, Perplexity)?

### RQ3: Cross-Domain Variance
**Question:** Do hallucination signatures differ significantly across domains (Medicine, Math, Finance, IS, Psychology)?

---

## 🎯 What's Been Implemented

### For RQ1: Feature Hypothesis ✅

**Baselines Implemented:**
- ✅ Naive Confidence (MaxProb + Perplexity)
- ✅ Semantic Uncertainty (Semantic Entropy + Energy)
- ✅ Context Only (Knowledge Popularity + Complexity)
- ✅ Semantic + Context
- ✅ Full Model (All features)

**Analysis:**
- Ablation study comparing all feature combinations
- AUROC comparison across models
- Statistical significance testing

**Output:**
- `rq1_ablation_study.csv` - Complete comparison table
- Bar chart showing AUROC for each model

---

### For RQ2: Semantic vs Naive ✅

**Naive Confidence Baselines:**
- ✅ **MaxProb**: Maximum token probability (model's stated confidence)
- ✅ **Perplexity**: Token-level perplexity (exp(-mean(logprobs)))
- ✅ **Mean/Min Logprob**: Additional naive metrics

**Semantic Uncertainty Metrics:**
- ✅ **Semantic Entropy**: Inconsistency across meaning (NLI clustering)
- ✅ **Semantic Energy**: Logit distribution analysis

**Analysis:**
- Direct AUROC comparison: Semantic vs Naive
- Improvement calculation (absolute and relative)
- Statistical test for significance

**Output:**
- `rq2_semantic_vs_naive.csv` - Detailed comparison
- `rq2_semantic_vs_naive.pdf` - Visual comparison chart
- Improvement metrics with significance test

---

### For RQ3: Cross-Domain Variance ✅

**Domain-Specific Analysis:**
- ✅ Train separate models for each domain
- ✅ Compare AUROC across domains
- ✅ Feature importance per domain
- ✅ Statistical test for domain differences (Chi-square)
- ✅ Pairwise domain comparisons
- ✅ Feature importance variation analysis

**Analysis:**
- Chi-square test for hallucination rate differences
- Domain-specific model performance
- Feature importance heatmap across domains
- Coefficient of variation for each feature
- Identification of domain-dependent features

**Output:**
- `rq3_domain_metrics.csv` - Performance by domain
- `rq3_feature_importance_differences.csv` - Feature variation
- `rq3_domain_auroc.pdf` - AUROC comparison chart
- `rq3_domain_feature_heatmap.pdf` - Feature importance heatmap
- Domain-specific models saved separately

---

## 🚀 How to Run

### Step 1: Process All Domains

```bash
# Process each domain individually
python run_pipeline.py --domain math
python run_pipeline.py --domain medicine --limit 5000  # Sample large dataset
python run_pipeline.py --domain finance
python run_pipeline.py --domain is_agents
python run_pipeline.py --domain psychology
```

### Step 2: Combine Features

```bash
# Combine all domain features into one file
python -c "
import pandas as pd
from pathlib import Path

dfs = []
for domain in ['math', 'medicine', 'finance', 'is_agents', 'psychology']:
    path = Path(f'data/features/{domain}_features.csv')
    if path.exists():
        df = pd.read_csv(path)
        dfs.append(df)
        print(f'Loaded {domain}: {len(df)} samples')

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('data/features/all_features.csv', index=False)
    print(f'\n✓ Combined: {len(combined)} samples across {len(dfs)} domains')
    print(f'Saved to: data/features/all_features.csv')
"
```

### Step 3: Run Comprehensive Analysis

```bash
# Analyze all three research questions
python analyze_research_questions.py \
  --features data/features/all_features.csv \
  --output-dir outputs/research_questions
```

**Expected time:** 15-30 minutes  
**Expected output:** Complete analysis for all RQs

---

## 📊 Expected Outputs

### Directory Structure

```
outputs/research_questions/
├── rq1_ablation_study.csv              # RQ1: Feature comparison
├── rq2_semantic_vs_naive.csv           # RQ2: Semantic vs Naive
├── rq3_domain_metrics.csv              # RQ3: Domain performance
├── rq3_feature_importance_differences.csv  # RQ3: Feature variation
├── figures/
│   ├── rq2_semantic_vs_naive.pdf       # RQ2 visualization
│   ├── rq2_semantic_vs_naive.png
│   ├── rq3_domain_auroc.pdf            # RQ3: Performance by domain
│   ├── rq3_domain_auroc.png
│   ├── rq3_domain_feature_heatmap.pdf  # RQ3: Feature importance heatmap
│   └── rq3_domain_feature_heatmap.png
└── domain_models/
    ├── xgboost_math.pkl                # Domain-specific models
    ├── xgboost_medicine.pkl
    ├── feature_importance_math.csv     # Domain-specific importance
    └── ...
```

### Key Results Files

**RQ1: `rq1_ablation_study.csv`**
```csv
model,num_features,auroc,accuracy,precision,recall,f1,ece
Baseline: Naive Confidence,4,0.6500,0.6200,...
Semantic Uncertainty,5,0.7800,0.7400,...
Context Only,12,0.7200,0.6900,...
Semantic + Context,17,0.8500,0.8100,...
Full Model (All Features),25,0.8700,0.8300,...
```

**RQ2: `rq2_semantic_vs_naive.csv`**
```csv
metric,naive_confidence,semantic_uncertainty,improvement,improvement_pct,hypothesis_supported
AUROC,0.6500,0.7800,0.1300,20.0,true
```

**RQ3: `rq3_domain_metrics.csv`**
```csv
domain,auroc,accuracy,precision,recall,f1,n_samples
Math,0.8900,0.8500,...
Medicine,0.8200,0.7800,...
Finance,0.7500,0.7200,...
IS,0.8100,0.7700,...
Psychology,0.7800,0.7400,...
```

---

## 📈 Interpretation Guide

### RQ1: Feature Hypothesis

**Success Criteria:**
- ✅ Full Model AUROC > Baseline by ≥0.05
- ✅ Semantic + Context > Individual components

**For Thesis:**
> "The full hybrid model achieved AUROC of X.XX, outperforming the naive baseline (X.XX) by X.XX points, demonstrating the value of combining semantic uncertainty with contextual features."

### RQ2: Semantic vs Naive

**Success Criteria:**
- ✅ Semantic Uncertainty AUROC > Naive Confidence by ≥0.05
- ✅ Improvement is statistically significant

**For Thesis:**
> "Semantic Entropy, which measures inconsistency across meaning, achieved AUROC of X.XX compared to naive confidence metrics (X.XX), representing a X% improvement. This supports our hypothesis that meaning-based uncertainty provides more reliable hallucination detection than token-level confidence."

### RQ3: Cross-Domain Variance

**Success Criteria:**
- ✅ Chi-square test p-value < 0.05 (domains differ)
- ✅ Domain-specific AUROC varies by ≥0.10
- ✅ At least 5 features show high cross-domain variation (CV > 0.3)

**For Thesis:**
> "Chi-square analysis revealed significant differences in hallucination patterns across domains (χ²=X.XX, p<0.001). Domain-specific models showed AUROC ranging from X.XX (Domain A) to X.XX (Domain B). Feature importance analysis identified X features with high cross-domain variation, indicating domain-dependent hallucination signatures."

---

## 🎓 For Your Thesis

### Results Section Structure

**Section 4.1: Overall Performance (RQ1)**
- Table: Ablation study results
- Figure: Bar chart of AUROC by model
- Text: Interpretation of hybrid approach

**Section 4.2: Semantic vs Naive Confidence (RQ2)**
- Table: Direct comparison
- Figure: Side-by-side AUROC comparison
- Text: Evidence for semantic superiority

**Section 4.3: Cross-Domain Analysis (RQ3)**
- Table: Domain-specific performance
- Figure 1: AUROC by domain
- Figure 2: Feature importance heatmap
- Text: Domain-dependent patterns

**Section 4.4: Feature Importance**
- Table: Top 15 features globally
- Figure: Feature importance with confidence intervals
- Text: Answer to "which features correlate with hallucinations?"

---

## 💡 Quick Test

Test with math domain first:

```bash
# Process math
python run_pipeline.py --domain math --limit 500

# Run RQ analysis (will work with single domain too)
python analyze_research_questions.py \
  --features data/features/math_features.csv \
  --output-dir outputs/research_questions_test
```

This will show you the format and validate everything works before processing all domains.

---

## 📋 Checklist

- [ ] Process all 5 domains
- [ ] Combine features into `all_features.csv`
- [ ] Run `analyze_research_questions.py`
- [ ] Review outputs in `outputs/research_questions/`
- [ ] Copy figures to thesis document
- [ ] Write results section based on outputs

**All three research questions will be answered comprehensively!** 🎓✨
