# Archived Utility Scripts

**Archive Date:** February 14, 2026  
**Reason:** These scripts were useful during development but are no longer needed for thesis work

## Archived Scripts (4)

### 1. `analyze_research_questions.py` (Feb 7)
**Purpose:** Old version of research questions analysis  
**Why Archived:** Superseded by newer modular scripts:
- `per_domain_ablation.py` - More rigorous ablation study
- `statistical_tests.py` - Proper hypothesis testing
- `generate_thesis_figures.py` - Better visualizations

**Status:** ⚠️ Old methodology, replaced by better approach

---

### 2. `generate_clean_viz.py` (Feb 7)
**Purpose:** Early visualization generation attempt  
**Why Archived:** Superseded by:
- `generate_thesis_figures.py` - Publication-quality figures
- `generate_domain_figures.py` - Per-domain visualizations

**Status:** ⚠️ Early prototype, replaced by final version

---

### 3. `inspect_datasets.py` (Feb 3)
**Purpose:** Debug tool to inspect dataset statistics  
**Why Archived:** Used only during initial data exploration

**Status:** ✅ Debugging utility, no longer needed

**What it did:**
- Counted samples per dataset
- Showed hallucination rates
- Displayed data quality metrics

---

### 4. `reprocess_cached_responses.py` (Feb 14)
**Purpose:** One-time script to reprocess cached API responses  
**Why Archived:** Specific fix for a one-time issue

**Status:** ✅ One-time use, task completed

**What it did:**
- Fixed cached response format issues
- Regenerated feature CSV files
- Updated response metadata

---

## Active Scripts (9)

The following scripts remain in `scripts/` and are used for thesis:

### Thesis Analysis (7 scripts)
1. `per_domain_ablation.py` - Per-domain ablation study (RQ1)
2. `aggregate_ablation_results.py` - Combines ablation results
3. `statistical_tests.py` - Hypothesis testing (all RQs)
4. `generate_thesis_figures.py` - 5 main thesis figures
5. `generate_domain_figures.py` - Per-domain visualizations
6. `train_consistent_models.py` - Trains final 5 models
7. `verify_consistency.py` - Verifies AUROC consistency

### Data Preparation (2 scripts)
8. `prepare_datasets.py` - Consolidates raw datasets
9. `convert_tatqa.py` - Converts TAT-QA JSON to CSV

---

## Can These Be Deleted?

**Yes, safely!** These scripts are not used in:
- ❌ The main pipeline (`run_pipeline.py`)
- ❌ Thesis figure generation
- ❌ Research question analysis
- ❌ Model training or evaluation

**Keep only for:** Historical reference of development process

---

## Evolution Timeline

```
Feb 3:  inspect_datasets.py created (initial exploration)
Feb 7:  analyze_research_questions.py created (first RQ attempt)
Feb 7:  generate_clean_viz.py created (early viz)
Feb 14: Newer scripts created (better methodology)
Feb 14: Archived old scripts (cleanup)
```

**Current Status:** All superseded by better implementations
