# Archived Old Datasets

**Archive Date:** February 14, 2026  
**Reason:** Superseded by better/larger datasets

## Archived Files (1)

### `financebench_sample_150.csv` (Feb 3)
**Purpose:** Initial finance dataset for testing  
**Why Archived:** Too small (only 150 samples), replaced by TAT-QA dataset

**Original Dataset:** FinanceBench  
**Samples:** 150  
**Domain:** Finance (document-grounded QA)

**Replaced By:** TAT-QA dataset  
**New File:** `../tatqa.csv`  
**New Samples:** 23,349 (156× larger!)  
**Advantage:** Much larger dataset for better statistical power

---

## Evolution: Finance Domain Dataset

### Phase 1: FinanceBench (Archived)
- **File:** `financebench_sample_150.csv`
- **Samples:** 150
- **Used:** Feb 3-6 for initial testing
- **Issue:** Too small for robust analysis
- **Status:** ⚠️ Superseded by TAT-QA

### Phase 2: TAT-QA (Current)
- **Source Files:** `TATQA/tatqa_dataset_*.json`
- **Processed File:** `../tatqa.csv`
- **Samples:** 23,349 (full dataset)
- **Thesis Used:** 500 (stratified sample)
- **Conversion Script:** `scripts/convert_tatqa.py`
- **Status:** ✅ Final dataset for thesis

---

## Current Raw Datasets (5 domains)

Located in `data/raw/`:

| Domain | Dataset | File | Total Samples | Thesis Used |
|--------|---------|------|---------------|-------------|
| Math | GSM8K | gsm8k.csv | 8,792 | 542 |
| Medicine | Med-HALT | med_halt.csv | 39,590 | 500 |
| Finance | TAT-QA | tatqa.csv | 23,349 | 500 |
| IS Agents | HalluMix | hallumix.csv | 9,396 | 500 |
| Psychology | TruthfulQA | TruthfulQA.csv | 817 | 500 |

**Total available:** 81,944 samples  
**Total used in thesis:** 2,542 samples (stratified/sampled)

---

## Why Switch from FinanceBench to TAT-QA?

### FinanceBench Limitations:
- ❌ Only 150 samples (too small)
- ❌ Insufficient for statistical analysis
- ❌ Can't support domain-specific model training

### TAT-QA Advantages:
- ✅ 23,349 samples (156× larger)
- ✅ Same domain (finance, document-grounded)
- ✅ Better statistical power
- ✅ Allows for robust stratified sampling
- ✅ Similar task structure (table + text → answer)

---

## Can This Be Deleted?

**Yes, safely!** This file is:
- ❌ Not used by any scripts
- ❌ Not referenced in thesis
- ❌ Superseded by TAT-QA dataset

**Keep only for:** Historical reference showing dataset evolution

---

## Timeline

```
Feb 3:  financebench_sample_150.csv added (initial testing)
Feb 5:  Identified need for larger finance dataset
Feb 6:  Switched to TAT-QA dataset
Feb 13: Converted TAT-QA JSON to CSV (23,349 samples)
Feb 13: Processed finance domain with TAT-QA data (500 stratified samples)
Feb 14: Archived FinanceBench (cleanup)
```

**Current Status:** Superseded by TAT-QA dataset
