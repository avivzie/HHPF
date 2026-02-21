# Scripts Directory

**Purpose:** Analysis scripts for thesis research questions and figure generation

## Overview

This directory contains **9 essential scripts** used to:
- Answer all 3 research questions with statistical rigor
- Generate publication-quality figures
- Train and validate models
- Prepare datasets

---

## Quick Reference

### 🎯 Main Thesis Workflow

```bash
# Step 1: Prepare datasets (one-time)
python scripts/prepare_datasets.py
python scripts/convert_tatqa.py  # For finance domain

# Step 2: Run main pipeline for each domain
python run_pipeline.py --domain math
python run_pipeline.py --domain medicine --limit 5000
python run_pipeline.py --domain finance
python run_pipeline.py --domain is_agents
python run_pipeline.py --domain psychology

# Step 3: Train consistent models
python scripts/train_consistent_models.py

# Step 4: Generate all thesis results
python scripts/per_domain_ablation.py
python scripts/aggregate_ablation_results.py
python scripts/statistical_tests.py

# Step 5: Generate all figures
python scripts/generate_domain_figures.py
python scripts/generate_thesis_figures.py

# Step 6: Verify consistency
python scripts/verify_consistency.py
```

---

## Scripts by Category

### 📊 Thesis Analysis Scripts (7)

#### 1. `per_domain_ablation.py`
**Purpose:** Per-domain ablation study for RQ1 and RQ2

**What it does:**
- Trains 5 feature subsets per domain (Naive, Semantic, Context, Semantic+Context, Full)
- Evaluates each subset with cross-validation
- Generates per-domain ablation results

**Output:**
- `outputs/ablation/{domain}_ablation_results.csv`
- `outputs/ablation/{domain}_feature_importance.csv`

**Usage:**
```bash
python scripts/per_domain_ablation.py
```

---

#### 2. `aggregate_ablation_results.py`
**Purpose:** Combines ablation results across all domains

**What it does:**
- Aggregates per-domain ablation results
- Calculates mean ± std AUROC across domains
- Prepares data for RQ1/RQ2 analysis

**Output:**
- `outputs/research_questions/rq1_rq2_aggregated_results.csv`
- `outputs/research_questions/per_domain_ablation_breakdown.csv`

**Usage:**
```bash
python scripts/aggregate_ablation_results.py
```

---

#### 3. `statistical_tests.py`
**Purpose:** Hypothesis testing for all 3 research questions

**What it does:**
- **RQ1:** Paired t-test (Hybrid vs Naive)
- **RQ2:** Paired t-test (Semantic vs Naive)
- **RQ3a:** Chi-square test (Hallucination rates)
- **RQ3c:** Feature importance variance analysis

**Output:**
- `outputs/research_questions/statistical_tests_summary.json`
- `outputs/research_questions/rq3a_hallucination_rates.csv`
- `outputs/research_questions/rq3c_feature_importance_variability.csv`

**Usage:**
```bash
python scripts/statistical_tests.py
```

---

#### 4. `generate_thesis_figures.py`
**Purpose:** Creates 5 main thesis figures (publication-quality)

**What it does:**
- RQ1: Ablation comparison (bar chart)
- RQ2: Semantic vs Naive (grouped bars)
- RQ3a: Hallucination rates (stacked bars)
- RQ3b: Domain AUROC (horizontal bars)
- RQ3c: Feature importance heatmap

**Output:**
- `outputs/research_questions/figures/rq1_ablation_comparison.pdf/png`
- `outputs/research_questions/figures/rq2_semantic_vs_naive.pdf/png`
- `outputs/research_questions/figures/rq3a_hallucination_rates.pdf/png`
- `outputs/research_questions/figures/rq3b_domain_auroc.pdf/png`
- `outputs/research_questions/figures/rq3c_feature_importance_heatmap.pdf/png`

**Usage:**
```bash
python scripts/generate_thesis_figures.py
```

---

#### 5. `generate_domain_figures.py`
**Purpose:** Creates per-domain visualizations

**What it does:**
- ROC curve with AUROC
- Accuracy-Rejection Curve (ARC)
- Calibration plot with ECE
- Confusion matrix
- Feature importance (top 15)

**Output:**
- `outputs/figures/{domain}/roc_curve_{domain}.pdf/png`
- `outputs/figures/{domain}/arc_{domain}.pdf/png`
- `outputs/figures/{domain}/calibration_{domain}.pdf/png`
- `outputs/figures/{domain}/confusion_matrix_{domain}.pdf/png`
- `outputs/figures/{domain}/feature_importance_{domain}.pdf/png`

**Usage:**
```bash
python scripts/generate_domain_figures.py
```

---

#### 6. `train_consistent_models.py`
**Purpose:** Trains final 5 models with guaranteed consistency

**What it does:**
- Loads cached features (no API calls)
- Trains all 5 domains with identical config
- Trains all 5 feature subsets for ablation
- Calculates comprehensive metrics
- Saves models and results

**Output:**
- `outputs/models/xgboost_{domain}.pkl` (5 models)
- `outputs/results/metrics_{domain}.json` (5 files)
- Ablation results for each domain

**Usage:**
```bash
python scripts/train_consistent_models.py
```

**Key Feature:** Ensures RQ3b AUROC values match individual domain ROC curves

---

#### 7. `verify_consistency.py`
**Purpose:** Quality assurance - verifies AUROC consistency

**What it does:**
- Checks that RQ3b domain AUROC matches saved model metrics
- Verifies individual ROC curve AUROC matches thesis figures
- Ensures no discrepancies across outputs

**Output:**
- Console report with verification status
- Highlights any inconsistencies > 0.0001 threshold

**Usage:**
```bash
python scripts/verify_consistency.py
```

**Expected:** ✅ All checks pass (max diff: 0.000000)

---

### 🔧 Data Preparation Scripts (2)

#### 8. `prepare_datasets.py`
**Purpose:** Consolidates and prepares raw datasets

**What it does:**
- Loads all 5 raw datasets
- Standardizes format (question, answer, context, ground_truth)
- Creates unified CSV files
- Validates data quality

**Output:**
- `data/raw/gsm8k_prepared.csv` (Math)
- `data/raw/medhalt_prepared.csv` (Medicine)
- `data/raw/tatqa_prepared.csv` (Finance)
- `data/raw/hallumix_is_agents_prepared.csv` (IS Agents)
- `data/raw/truthfulqa_prepared.csv` (Psychology)

**Usage:**
```bash
python scripts/prepare_datasets.py
```

---

#### 9. `convert_tatqa.py`
**Purpose:** Converts TAT-QA JSON to CSV format

**What it does:**
- Parses TAT-QA JSON files
- Extracts questions, tables, and gold answers
- Converts tables to text representation
- Creates standardized CSV

**Output:**
- `data/raw/tatqa_prepared.csv`

**Usage:**
```bash
python scripts/convert_tatqa.py
```

**Note:** This is finance-domain specific

---

## Dependencies

All scripts use modules from `src/`:
- `src/features/` - Feature extraction
- `src/classifier/` - XGBoost training
- `src/evaluation/` - Metrics calculation
- `src/data_preparation/` - Data loading
- `src/utils.py` - Utility functions

**No duplication** - Scripts orchestrate, `src/` provides functionality

---

## Archived Scripts

Old/superseded scripts are in:
- `scripts/archive_old_utilities/` - Development utilities (not needed for thesis)

---

## Expected Runtime

| Script | Time | Cost | Notes |
|--------|------|------|-------|
| `prepare_datasets.py` | 2 min | Free | One-time |
| `convert_tatqa.py` | 1 min | Free | One-time |
| `train_consistent_models.py` | 10 min | Free | Uses cached features |
| `per_domain_ablation.py` | 15 min | Free | Cross-validation |
| `aggregate_ablation_results.py` | 10 sec | Free | - |
| `statistical_tests.py` | 30 sec | Free | - |
| `generate_thesis_figures.py` | 1 min | Free | - |
| `generate_domain_figures.py` | 2 min | Free | - |
| `verify_consistency.py` | 10 sec | Free | - |

**Total:** ~30 minutes (after pipeline runs complete)

---

## Outputs Summary

### Research Questions
- `outputs/research_questions/` - All RQ results and figures

### Models
- `outputs/models/` - 5 final trained models

### Metrics
- `outputs/results/` - Per-domain metrics JSON files

### Figures
- `outputs/figures/{domain}/` - Per-domain visualizations
- `outputs/research_questions/figures/` - Thesis figures

### Ablation
- `outputs/ablation/` - Per-domain ablation results

---

## Quality Assurance

All scripts use:
- ✅ Fixed random seeds (`random_state=42`)
- ✅ Identical XGBoost configuration
- ✅ Stratified train/test splits
- ✅ Comprehensive logging
- ✅ Error handling

**Result:** Fully reproducible thesis outputs

---

## For More Information

- **Project overview:** `START_HERE.md`
- **Research questions:** `RESEARCH_QUESTIONS.md`
- **Full documentation:** `README.md`
- **Script audit:** `scripts/SCRIPTS_AUDIT.md`
