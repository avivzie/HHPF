# CRITICAL BUG: Train/Test Distribution Mismatch
**Date:** 2026-02-07  
**Severity:** 🚨 CRITICAL - Invalidates all 500-sample results  
**Status:** Root cause identified

---

## The Smoking Gun

### Psychology: 26.5 Percentage Point Train/Test Gap
```
Train: 54.5% hallucinations (218/400)
Test:  28.0% hallucinations (28/100)
Gap:   26.5 percentage points ❌
```

**Why this breaks everything:**
- Model trains on data where ~55% are hallucinations
- Model tests on data where only ~28% are hallucinations
- **Different distributions = model can't generalize**
- AUROC collapses to ~0.53 (near-random)

### Medicine: Extreme Class Imbalance in Test
```
Train: 90.2% hallucinations (361/400, 39 faithful)
Test:  94.0% hallucinations (94/100, only 6 faithful!)
Gap:   3.8 percentage points (acceptable)
```

**Why this breaks evaluation:**
- Only **6 faithful samples** in test set
- AUROC becomes unstable with so few positive examples
- Random variation in those 6 samples dominates performance
- Results not statistically meaningful

---

## Root Cause: Stratified Splitting Failure

The `train_test_split` with `stratify=y` should ensure:
- Same hallucination rate in train and test
- Proportional representation of both classes

**But it failed catastrophically for psychology.**

### Why Stratification Failed (Psychology)

**Hypothesis:** The stratification happens AFTER labeling all 500 samples, but:
1. First 100 samples (cached) labeled with high hallucination rate
2. Next 400 samples labeled with low hallucination rate  
3. Random split puts more "low hallucination" samples in test
4. Result: 54.5% train vs 28.0% test

**Alternative:** Stratification itself has a bug or random seed issue.

---

## Comparison: Working vs Broken Runs

| Run | Samples | Train Hall% | Test Hall% | Gap | AUROC | Status |
|-----|---------|-------------|------------|-----|-------|--------|
| Psychology 100 | 100 | 83.8% | 80.0% | 3.8% | 0.75 | ✅ Working |
| **Psychology 500** | **500** | **54.5%** | **28.0%** | **26.5%** | **0.53** | **❌ Broken** |
| Medicine 75 | 75 | - | - | - | 0.64 | ✅ Working |
| **Medicine 500** | **500** | **90.2%** | **94.0%** | **3.8%** | **0.51** | **❌ Broken** |

**Pattern:**
- Small samples: Train/test match, good AUROC
- Large samples: Train/test mismatch OR extreme imbalance, poor AUROC

---

## The Bug: Where It Lives

### Suspect Code: `src/data_preparation/process_datasets.py`

The stratified splitting logic for medicine was added but may have issues:

```python
# Stratified split for medicine
if args.domain == 'medicine':
    # Create temporary stratification label
    df['_stratify_label'] = df['ground_truth'].apply(
        lambda x: 'none' if 'None of the above' in str(x) else 'specific'
    )
    
    # Split with stratification
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['_stratify_label']  # ❌ Wrong variable!
    )
```

**BUG:** We stratify on `ground_truth` type (none vs specific), not on `hallucination_label`!

### Psychology Has No Stratification
Psychology domain has no special handling in `process_datasets.py`, so it uses default splitting which doesn't stratify on labels at all.

---

## Fix Required

### Option 1: Stratify on Actual Labels (RECOMMENDED)
```python
# For ALL domains
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['hallucination_label']  # ✅ Stratify on actual target!
)
```

**Problem:** Labels don't exist yet at split time in current pipeline!

### Option 2: Stratify After Labeling
Restructure pipeline:
1. Load dataset
2. Generate responses (or load from cache)
3. **Label all responses first**
4. **Then split train/test** with stratification on labels
5. Extract features
6. Train model

### Option 3: Post-Split Balancing
After splitting, manually balance train/test to have similar distributions.

---

## Impact on Results

### All 500-Sample Results Invalid
- **Medicine 500 (RUN-008):** AUROC 0.44 - INVALID (only 6 test faithful)
- **Medicine 500 (RUN-012):** AUROC 0.51 - INVALID (only 6 test faithful)
- **Psychology 500 (RUN-013):** AUROC 0.53 - INVALID (26.5% train/test gap)

### Valid Results Remain
- **Math 542:** AUROC 0.79 - VALID ✅
- **Finance 150:** AUROC 0.68 - VALID ✅  
- **Psychology 100:** AUROC 0.75 - VALID ✅
- **Medicine 75:** AUROC 0.64 - VALID ✅

---

## Action Plan

### Immediate (Today)
1. ✅ Bug identified and documented
2. 🔧 Fix stratification in `process_datasets.py`
3. 🧪 Test fix on small sample (50 samples)
4. 📊 Verify train/test distributions match

### Short-term (This Week)
1. Re-run medicine 500 with fixed stratification
2. Re-run psychology 500 with fixed stratification
3. Validate results (AUROC should improve to 0.60-0.70 range)
4. Document fix in thesis

### Thesis Strategy
- Acknowledge bug discovery in methodology
- Show before/after results
- Discuss importance of proper stratification
- Use as learning/contribution point

---

## Lessons Learned

1. **Always verify train/test distributions** - Check hallucination rates match
2. **Stratify on target variable** - Not on proxy variables
3. **Small-sample success doesn't guarantee large-sample success** - Scale reveals bugs
4. **Extreme class imbalance breaks evaluation** - Need minimum samples per class
5. **Manual inspection catches bugs** - Diagnostic checks essential

---

**Bottom Line:** Both 500-sample runs are INVALID due to train/test distribution mismatch. The bug is in the splitting logic (not stratifying on actual labels). Fix required before trusting any large-scale results.
