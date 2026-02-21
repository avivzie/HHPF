# Scripts Directory Audit

**Date:** February 14, 2026  
**Purpose:** Identify script purposes and potential consolidation opportunities

## Current Scripts (14 total)

### 📊 Category 1: Thesis Analysis Scripts (KEEP - Active Use)

These scripts generate the final thesis results and are essential:

| Script | Purpose | Status | Used By |
|--------|---------|--------|---------|
| `per_domain_ablation.py` | Per-domain ablation study (RQ1) | ✅ **KEEP** | Thesis RQ1 |
| `aggregate_ablation_results.py` | Combines ablation results across domains | ✅ **KEEP** | Thesis RQ1/RQ2 |
| `statistical_tests.py` | Hypothesis testing for all 3 RQs | ✅ **KEEP** | Thesis RQ1/RQ2/RQ3 |
| `generate_thesis_figures.py` | Creates 5 main thesis figures | ✅ **KEEP** | Thesis figures |
| `generate_domain_figures.py` | Creates per-domain ROC/ARC/calibration plots | ✅ **KEEP** | Thesis figures |
| `train_consistent_models.py` | Trains final 5 models with consistency | ✅ **KEEP** | Final models |
| `verify_consistency.py` | Verifies AUROC consistency across outputs | ✅ **KEEP** | Quality assurance |

**Total: 7 scripts** - All actively used in thesis

---

### 🔧 Category 2: Data Preparation Scripts (KEEP - Setup)

These scripts prepare datasets before running the pipeline:

| Script | Purpose | Status | Used By |
|--------|---------|--------|---------|
| `prepare_datasets.py` | Consolidates and prepares raw datasets | ✅ **KEEP** | Initial setup |
| `convert_tatqa.py` | Converts TAT-QA JSON to CSV format | ✅ **KEEP** | Finance domain |

**Total: 2 scripts** - Needed for data setup

---

### 🔍 Category 3: Utility/Debug Scripts (ARCHIVE - Optional)

These scripts were useful during development but not essential for thesis:

| Script | Purpose | Status | Recommendation |
|--------|---------|--------|----------------|
| `inspect_datasets.py` | Inspects dataset statistics | ⚠️ **ARCHIVE** | Debugging only |
| `analyze_research_questions.py` | Old RQ analysis (superseded) | ⚠️ **ARCHIVE** | Replaced by newer scripts |
| `generate_clean_viz.py` | Early visualization attempt | ⚠️ **ARCHIVE** | Superseded by generate_thesis_figures.py |
| `reprocess_cached_responses.py` | Reprocesses API responses | ⚠️ **ARCHIVE** | One-time use |

**Total: 4 scripts** - Can be archived

---

## Comparison: `scripts/` vs `src/`

### No Duplication - Different Purposes

**`src/` modules** (Library code):
- Reusable functions and classes
- Used by multiple scripts
- Examples: feature extraction, model training, data loading
- Purpose: **Building blocks**

**`scripts/` files** (Analysis scripts):
- End-to-end workflows
- Specific analyses (ablation study, statistical tests)
- Generate thesis outputs
- Purpose: **Orchestration and analysis**

### Key Differences

| Aspect | `src/` | `scripts/` |
|--------|--------|------------|
| Role | Library/modules | Executable scripts |
| Imports | From each other | From `src/` modules |
| Reusability | High | Low (specific analyses) |
| Testing | Unit testable | Integration scripts |
| Examples | `src/features/epistemic_uncertainty.py` | `scripts/per_domain_ablation.py` |

**No consolidation needed** - They serve different purposes!

---

## Main Entry Points

### 1. Main Pipeline (Uses `src/`)
```bash
python run_pipeline.py --domain math
```
**Location:** Root directory  
**Uses:** All `src/` modules  
**Purpose:** Process raw data → features → training → evaluation

### 2. Thesis Analysis Scripts (Use `src/` + pipeline outputs)
```bash
python scripts/per_domain_ablation.py
python scripts/statistical_tests.py
python scripts/generate_thesis_figures.py
```
**Location:** `scripts/` directory  
**Uses:** `src/` modules + cached features/models  
**Purpose:** Answer RQs, generate thesis figures

---

## Recommendations

### ✅ Keep These (9 scripts)

**Essential for thesis:**
1. `per_domain_ablation.py`
2. `aggregate_ablation_results.py`
3. `statistical_tests.py`
4. `generate_thesis_figures.py`
5. `generate_domain_figures.py`
6. `train_consistent_models.py`
7. `verify_consistency.py`

**Essential for setup:**
8. `prepare_datasets.py`
9. `convert_tatqa.py`

### 📦 Archive These (4 scripts)

Move to `scripts/archive_old_utilities/`:

1. `inspect_datasets.py` - Debugging tool
2. `analyze_research_questions.py` - Old version (superseded)
3. `generate_clean_viz.py` - Early viz attempt (superseded)
4. `reprocess_cached_responses.py` - One-time fix

### 🗑️ Delete These (1 file)

- `export_diagrams.sh` - Shell script, should be documented in README

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   HHPF Workflow                          │
└─────────────────────────────────────────────────────────┘

Step 1: Data Preparation (One-time)
├── scripts/prepare_datasets.py      → data/raw/*.csv
└── scripts/convert_tatqa.py         → data/raw/tatqa_*.csv

Step 2: Main Pipeline (Per domain)
├── run_pipeline.py                  
│   ├── Uses: src/data_preparation/  → Load & label data
│   ├── Uses: src/inference/         → Generate responses
│   ├── Uses: src/features/          → Extract features
│   ├── Uses: src/classifier/        → Train model
│   └── Uses: src/evaluation/        → Evaluate model
└── Output: data/features/{domain}_features.csv

Step 3: Thesis Analysis (After all domains)
├── scripts/train_consistent_models.py    → outputs/models/*.pkl
├── scripts/per_domain_ablation.py        → RQ1/RQ2 results
├── scripts/aggregate_ablation_results.py → Combined results
├── scripts/statistical_tests.py          → p-values, effect sizes
├── scripts/generate_domain_figures.py    → Per-domain figures
├── scripts/generate_thesis_figures.py    → 5 main figures
└── scripts/verify_consistency.py         → QA check

Step 4: Thesis Writing
└── Use all outputs/ files in dissertation
```

---

## Key Insight

**There is NO duplication between `scripts/` and `src/`:**

- `src/` = Reusable library code (the "engine")
- `scripts/` = Analysis workflows (the "driver")
- Scripts **use** src modules, they don't duplicate them

**Example:**
```python
# src/features/epistemic_uncertainty.py
def compute_semantic_entropy(responses):
    # Reusable function
    ...

# scripts/per_domain_ablation.py
from src.features.epistemic_uncertainty import compute_semantic_entropy
# Uses the function in a specific analysis
```

---

## Action Items

### Option 1: Clean Archive (Recommended)
```bash
mkdir scripts/archive_old_utilities
mv scripts/inspect_datasets.py scripts/archive_old_utilities/
mv scripts/analyze_research_questions.py scripts/archive_old_utilities/
mv scripts/generate_clean_viz.py scripts/archive_old_utilities/
mv scripts/reprocess_cached_responses.py scripts/archive_old_utilities/
rm scripts/export_diagrams.sh
```

### Option 2: Keep Everything (Safe)
- Leave as-is, document in this audit
- Old scripts won't hurt, just clutter

---

## Summary

✅ **No consolidation needed** - `src/` and `scripts/` serve different purposes  
✅ **9 essential scripts** - Used in thesis  
⚠️ **4 optional scripts** - Can be archived  
🗑️ **1 shell script** - Can be deleted

**Conclusion:** Structure is good! Only minor cleanup recommended.
