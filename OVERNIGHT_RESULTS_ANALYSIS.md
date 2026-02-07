# Overnight Runs: Critical Performance Issues Discovered
**Date:** 2026-02-07  
**Status:** ⚠️ Both runs completed but show near-random performance  
**Severity:** HIGH - Requires immediate investigation

---

## Summary: Both Domains Failed at 500 Samples

| Domain | Samples | Test AUROC | Train AUROC | Status |
|--------|---------|------------|-------------|--------|
| Medicine | 500 | **0.5089** | 0.9914 | ❌ Near-random |
| Psychology | 500 | **0.5312** | 0.9987 | ❌ Near-random |
| Psychology | 100 | 0.7500 | 0.9690 | ✅ Previously working |
| Medicine | 75 | 0.6364 | - | ✅ Previously working |

**Both domains show:**
1. 🚨 Near-random test performance (AUROC ~0.50)
2. 🚨 Extreme overfitting (train AUROC ~0.99, test ~0.51)
3. 🚨 Performance degradation from smaller sample sizes

---

## RUN-012: Medicine 500-Sample Results

### Performance Metrics
- **Test AUROC: 0.5089** (barely better than random 0.5)
- Test Accuracy: 0.7600
- Test ECE: 0.2425
- Train AUROC: 0.9914 (extreme overfitting)

### Class Distribution
- Train: 90.2% hallucinations (361/400)
- Test: 94.0% hallucinations (94/100)
- **Distribution appears consistent between train/test**

### Comparison with Previous Runs
| Run | Samples | AUROC | Notes |
|-----|---------|-------|-------|
| RUN-006 (75 samples) | 75 | 0.6364 | Working |
| RUN-008 (500 #1) | 500 | 0.4397 | Poor |
| **RUN-012 (500 #3)** | **500** | **0.5089** | **Still poor** |

### Critical Findings
1. **Reproducible poor performance** - Third 500-sample run confirms pattern
2. **Not a labeling distribution issue** - 90.2% vs 94.0% is reasonable
3. **Extreme overfitting** - Model memorizing training data (0.99 train AUROC)
4. **Scale effect confirmed** - 75 samples work (0.64), 500 samples fail (0.51)

---

## RUN-013: Psychology 500-Sample Results

### Performance Metrics
- **Test AUROC: 0.5312** (barely better than random)
- Test Accuracy: 0.5200
- Test ECE: 0.2588
- Train AUROC: 0.9987 (extreme overfitting)

### Class Distribution
- Train: 54.5% hallucinations (218/400)
- Test: 28.0% hallucinations (28/100)
- **🚨 MAJOR TRAIN/TEST DISTRIBUTION SHIFT**

### Comparison with Previous Runs
| Run | Samples | Train Hall% | Test Hall% | AUROC | Notes |
|-----|---------|-------------|------------|-------|-------|
| RUN-010 (100 samples) | 100 | 83.8% | 80.0% | 0.7500 | Working ✅ |
| **RUN-013 (500 samples)** | **500** | **54.5%** | **28.0%** | **0.5312** | Failed ❌ |

### Critical Findings
1. **Massive label distribution change** - 83.8% → 54.5% hallucinations in training
2. **Severe train/test mismatch** - 54.5% train vs 28.0% test (26.5% gap!)
3. **Performance collapsed** - From 0.75 to 0.53 AUROC
4. **Labeling inconsistency suspected** - Different samples labeled differently at scale

---

## Root Cause Analysis

### Why Both Domains Failed at 500 Samples

**Common Pattern:**
- Both domains: Good performance at small scale (75-100 samples)
- Both domains: Near-random at 500 samples
- Both domains: Extreme overfitting (train AUROC ~0.99)

**Possible Causes:**

#### 1. Train/Test Distribution Mismatch (HIGH PROBABILITY)
**Evidence from Psychology:**
- Train: 54.5% hallucinations
- Test: 28.0% hallucinations
- 26.5 percentage point gap

**Why this breaks models:**
- Model learns patterns from majority-hallucination training data
- Test set has minority-hallucination distribution
- Features that predict training labels don't generalize

**Likely source:**
- Stratified splitting may be failing
- Dataset not shuffled properly before split
- Cached samples (100) vs new samples (400) have different characteristics
- `train_test_split` stratification not working as expected

#### 2. Labeling Logic Changes at Scale (HIGH PROBABILITY - Psychology)
**Evidence from Psychology:**
- 100 samples: 83.8% hallucinations (consistent with TruthfulQA adversarial design)
- 500 samples: 54.5% hallucinations (much lower - suspicious)

**Why this could happen:**
- Semantic similarity threshold (0.7) too lenient for some answer types
- Different answer patterns in later samples (501-817 in TruthfulQA)
- Model caching issues causing inconsistent labeling
- Ground truth quality varies across dataset

#### 3. Feature Quality Degradation (MODERATE PROBABILITY)
**Evidence:**
- All features have high NULL rates (semantic_energy 100% NULL)
- Only semantic_entropy available as strong signal
- May not generalize well with class imbalance

#### 4. Extreme Class Imbalance (Medicine) (MODERATE PROBABILITY)
**Evidence from Medicine:**
- 94% hallucinations in test set (only 6 faithful samples!)
- scale_pos_weight: 0.11 (severe imbalance)
- Hard to learn patterns from 6 positive examples

**Why this breaks evaluation:**
- AUROC unstable with very few positive examples
- Model defaults to majority class
- Random chance in small faithful sample placement

---

## Critical Issue: Psychology Label Inconsistency

### The Smoking Gun
**100-sample run (RUN-011):**
```
Train: 83.8% hallucinations, Test: 80.0% hallucinations
AUROC: 0.7500 ✅
```

**500-sample run (RUN-013):**
```
Train: 54.5% hallucinations, Test: 28.0% hallucinations
AUROC: 0.5312 ❌
```

### What Changed?
1. **Cached 100 samples reused** - These had 80%+ hallucination rate
2. **400 new samples generated** - These must have MUCH LOWER hallucination rate
3. **Result:** Training data becomes heterogeneous mix

### Hypothesis
The semantic similarity labeling (`SentenceTransformer` with 0.7 threshold) produces:
- **Higher hallucination rate** on first 100 samples (adversarial questions)
- **Lower hallucination rate** on samples 101-500 (different question types?)
- **Inconsistent labeling** across the dataset

---

## Investigation Required

### Immediate Checks (Priority 1)

**1. Verify Psychology Labeling Consistency:**
```bash
# Check labeling distribution in responses
python << EOF
import pandas as pd
df = pd.read_csv('data/features/psychology_train_responses.csv')
print("Labeling by sample range:")
print("Samples 0-99:", df.iloc[:100]['hallucination_label'].mean())
print("Samples 100-499:", df.iloc[100:]['hallucination_label'].mean())
print("\nAll samples:", df['hallucination_label'].mean())
EOF
```

**2. Check Train/Test Split Logic:**
```bash
# Verify stratification working
python << EOF
import pandas as pd
df = pd.read_csv('data/processed/psychology_processed.csv')
print("Split distribution:")
print(df.groupby(['split', 'hallucination_label']).size())
print("\nHallucination rates:")
print(df.groupby('split')['hallucination_label'].mean())
EOF
```

**3. Manual Sample Inspection (Psychology):**
- Inspect samples 1-100 vs 101-500
- Compare ground truths and responses
- Check if semantic similarity threshold makes sense

**4. Check Medicine Class Imbalance:**
```bash
# How many faithful samples in test?
python << EOF
import pandas as pd
df = pd.read_csv('data/processed/medicine_processed.csv')
test = df[df['split'] == 'test']
print(f"Test set size: {len(test)}")
print(f"Faithful: {(test['hallucination_label'] == 0).sum()}")
print(f"Hallucinations: {(test['hallucination_label'] == 1).sum()}")
EOF
```

### Deep Investigation (Priority 2)

**5. Compare Feature Distributions:**
- Train vs test semantic_entropy distributions
- Check if features discriminate differently at scale

**6. Test Different Train/Test Splits:**
- Try different random seeds
- Verify if test performance varies significantly

**7. Threshold Sensitivity Analysis (Psychology):**
- Test semantic similarity thresholds: 0.5, 0.6, 0.7, 0.8
- Find optimal threshold for consistency

---

## Immediate Next Steps

### Option A: Fix Psychology Labeling (RECOMMENDED)
1. Investigate why labeling changed at 500 samples
2. Manual inspection of samples 101-500
3. Adjust semantic similarity threshold if needed
4. Re-label all 500 samples consistently
5. Re-run with fixed labels

### Option B: Accept Small-Scale Results
1. Use 100-sample psychology results (0.75 AUROC)
2. Use 75-sample medicine results (0.64 AUROC)
3. Document scale limitations in thesis
4. Focus on other domains (math/finance working)

### Option C: Investigate Scale Effects Systematically
1. Run intermediate scales: 150, 250, 350 samples
2. Plot AUROC vs sample size
3. Identify where performance breaks
4. Document scale effect as research finding

---

## Comparison with Working Domains

| Domain | Samples | AUROC | Status | Notes |
|--------|---------|-------|--------|-------|
| **Math** | 542 | **0.7918** | ✅ Working | Baseline, no issues |
| **Finance** | 150 | **0.6827** | ✅ Working | Small scale, stable |
| Medicine | 75 | 0.6364 | ✅ Working | Small scale only |
| Medicine | 500 | 0.5089 | ❌ Failed | 3rd reproducible failure |
| Psychology | 100 | 0.7500 | ✅ Working | Small scale only |
| Psychology | 500 | 0.5312 | ❌ Failed | Label inconsistency |

**Pattern:** All domains work at small scale (75-150), medicine/psychology fail at 500.

---

## Thesis Implications

### Current State
- ✅ **Math domain validated** (542 samples, 0.79 AUROC)
- ✅ **Finance domain validated** (150 samples, 0.68 AUROC)
- ❌ **Medicine domain broken at scale** (500 samples, ~0.50 AUROC)
- ❌ **Psychology domain broken at scale** (500 samples, ~0.53 AUROC)
- ⏳ **IS/Agents domain pending**

### Research Questions Status
- **RQ1 (Semantic entropy effectiveness):** Can answer with math/finance
- **RQ2 (Domain variability):** ⚠️ Partially answerable, but scale issues complicate
- **RQ3 (Feature importance):** Can analyze math/finance

### Options for Thesis
1. **Focus on working domains** - Math + Finance provide evidence
2. **Document scale effects** - Honest finding about limitations
3. **Fix and re-run** - Invest time to debug medicine/psychology
4. **Accept small-scale results** - Use 75/100 sample results with caveats

---

## Recommended Action Plan

### Today (Immediate)
1. ✅ Run psychology labeling consistency check
2. ✅ Run medicine class imbalance check  
3. ✅ Manual inspection of 10 random psychology samples (samples 101-500)
4. 📊 Decide: Fix vs Accept vs Investigate further

### This Week
1. If fixable: Re-run with corrected labeling
2. If not fixable: Document as scale limitation
3. Validate IS/agents domain (start with 100 samples)
4. Analyze math/finance in depth

### Thesis Strategy
- Lead with strong results (math 0.79, finance 0.68)
- Document scale effects as finding
- Use working domains for main conclusions
- Discuss limitations transparently

---

**Bottom Line:** Both 500-sample runs failed with near-random performance. Psychology shows clear labeling inconsistency (83.8% → 54.5% hallucination rate). Medicine shows reproducible poor performance. Need to investigate and decide whether to fix, accept small-scale results, or document as scale limitation.
