# HHPF Results Directory

**Last Updated:** February 14, 2026  
**Status:** ✅ Research Complete - All 5 Domains Analyzed

---

## Navigation

This directory contains **per-domain metrics** from the full pipeline runs. For complete research analysis, see:

**Primary Results Location:** [`../research_questions/`](../research_questions/)
- Complete statistical analysis for all 3 research questions
- Publication-quality figures (10 PDFs/PNGs)
- Comprehensive 14-page results summary
- Statistical test results (p-values, effect sizes)

---

## Directory Structure

### Current Files (This Directory)

**Per-Domain Metrics (Full Model Results):**
- `metrics_math.json` - Math domain (GSM8K, n=542, AUROC 0.797)
- `metrics_is_agents.json` - IS Agents (HalluMix, n=500, AUROC 0.703)
- `metrics_psychology.json` - Psychology (TruthfulQA, n=500, AUROC 0.671)
- `metrics_medicine.json` - Medicine (Med-HALT, n=500, AUROC 0.619)
- `metrics_finance.json` - Finance (TAT-QA, n=500, AUROC 0.632)

**Archived Files:**
- `archive/` - Old summaries, intermediate validations, 3-domain comparisons

### Related Directories

**Ablation Studies:**
- **Location:** [`../ablation/`](../ablation/)
- **Contents:** Per-domain ablation results (5 feature subsets × 5 domains)
- **Files:** 10 CSVs (ablation results + feature importance)

**Research Questions:**
- **Location:** [`../research_questions/`](../research_questions/)
- **Contents:** Cross-domain analysis, statistical tests, figures
- **Key Files:**
  - `RESULTS_SUMMARY.md` - Complete results (14 pages)
  - `VALIDATION_CHECKLIST.md` - Methodology validation
  - `rq1_rq2_aggregated_results.csv` - Ablation study aggregated
  - `statistical_tests_summary.json` - All statistical tests
  - `figures/` - 10 publication-quality figures (PDF + PNG)
  - `DIAGRAM_EXPORT_GUIDE.md` - Instructions for exporting pipeline diagrams

**Pipeline Diagrams:**
- **Location:** [`../../docs/PIPELINE_DIAGRAMS.md`](../../docs/PIPELINE_DIAGRAMS.md)
- **Contents:** 8 mermaid diagrams showing complete A-Z workflow
- **Key Diagrams:**
  - Complete HHPF Technical Pipeline (6 stages)
  - Research Methodology Flow (Phase A → Phase B)
  - Feature Engineering Architecture
  - Ablation Study Design
  - Cross-Domain Statistical Analysis
  - Conceptual Overview & End-to-End Flow
- **Export Guide:** [`../research_questions/DIAGRAM_EXPORT_GUIDE.md`](../research_questions/DIAGRAM_EXPORT_GUIDE.md)

**Visualizations:**
- **Location:** [`../figures/`](../figures/)
- **Contents:** Per-domain figures (ROC, ARC, calibration, confusion matrix, feature importance)
- **Subdirectories:** `math/`, `is_agents/`, `psychology/`, `medicine/`, `finance/`

**Models:**
- **Location:** [`../models/`](../models/)
- **Contents:** Trained XGBoost models (one per domain)
- **Files:** `xgboost_math.pkl`, `xgboost_is_agents.pkl`, etc.

---

## Quick Reference: Final Results

### Per-Domain Performance (Full Model)

| Domain | Dataset | n_samples | AUROC | Accuracy | ECE | Hall. Rate |
|--------|---------|-----------|-------|----------|-----|------------|
| Math | GSM8K | 542 | 0.797 | 0.806 | 0.184 | 29.0% |
| IS Agents | HalluMix | 500 | 0.703 | 0.810 | 0.341 | 87.8% |
| Psychology | TruthfulQA | 500 | 0.671 | 0.740 | 0.344 | 25.0% |
| Finance | TAT-QA | 500 | 0.632 | 0.640 | 0.196 | 73.8% |
| Medicine | Med-HALT | 500 | 0.619 | 0.690 | 0.057 | 49.6% |
| **Mean±SD** | - | - | **0.684±0.071** | 0.737±0.078 | 0.224±0.115 | 53.0±28.6% |

**Performance Ranking:** Math (1st) > IS Agents (2nd) > Psychology (3rd) > Finance (4th) > Medicine (5th)

**Best Calibration:** Medicine (ECE 0.057)  
**Highest Accuracy:** IS Agents (0.810)

### Research Questions Summary

**RQ1: Do hybrid features outperform baselines?**
- **Result:** Partial support (p = 0.087, trend-level)
- **Hybrid AUROC:** 0.678 ± 0.055
- **Naive AUROC:** 0.562 ± 0.097
- **Improvement:** +0.116 (+20.7%)
- **Effect size:** Cohen's d = 1.007 (large)

**RQ2: Does semantic uncertainty outperform naive confidence?**
- **Result:** Not supported (p = 0.262)
- **Semantic AUROC:** 0.605 ± 0.076
- **Naive AUROC:** 0.562 ± 0.097
- **Improvement:** +0.043 (+7.7%)
- **Effect size:** Cohen's d = 0.312 (small-medium)

**RQ3: Do domains differ significantly?**
- **Result:** Strongly supported (p < 0.001)
- **RQ3a:** Hallucination rates differ significantly (χ² = 614.64, p < 0.001)
- **RQ3b:** AUROC variance observed (range 0.619-0.797)
- **RQ3c:** 26 out of 41 features (63%) show high cross-domain variation (CV > 0.3)

### Feature Ablation Results

| Feature Subset | n_features | AUROC (mean±std) | vs Baseline | Rank |
|----------------|------------|------------------|-------------|------|
| Full | 41-48 | 0.685 ± 0.071 | +0.123 (+21.9%) | 1st |
| Semantic+Context | 24-27 | 0.678 ± 0.055 | +0.116 (+20.7%) | 2nd |
| Semantic-Only | 3 | 0.605 ± 0.076 | +0.043 (+7.7%) | 3rd |
| Context-Only | 21-24 | 0.589 ± 0.043 | +0.027 (+4.8%) | 4th |
| Naive-Only | 4 | 0.562 ± 0.097 | baseline | 5th |

---

## For Thesis Writing

### Primary Reference Documents

1. **Main Results:** [`../research_questions/RESULTS_SUMMARY.md`](../research_questions/RESULTS_SUMMARY.md)
   - 14-page comprehensive analysis
   - All RQ answered with statistical rigor
   - Includes interpretation and implications

2. **Validation:** [`../research_questions/VALIDATION_CHECKLIST.md`](../research_questions/VALIDATION_CHECKLIST.md)
   - Methodology verification
   - Deliverables checklist
   - Data integrity confirmation

### Figures for Inclusion

All figures available in PDF (thesis) and PNG (presentations) formats:

**Location:** [`../research_questions/figures/`](../research_questions/figures/)

1. **Figure 1:** `rq1_ablation_comparison.pdf` - Feature ablation results with error bars
2. **Figure 2:** `rq2_semantic_vs_naive.pdf` - Per-domain comparison
3. **Figure 3:** `rq3a_hallucination_rates.pdf` - Stacked bar chart of rates
4. **Figure 4:** `rq3b_domain_auroc.pdf` - Domain performance comparison
5. **Figure 5:** `rq3c_feature_importance_heatmap.pdf` - 15 features × 5 domains

### Tables for Inclusion

**Location:** [`../research_questions/`](../research_questions/)

1. **Table 1:** `rq1_rq2_aggregated_results.csv` - Ablation study summary
2. **Table 2:** `per_domain_ablation_breakdown.csv` - 25 models (5 domains × 5 subsets)
3. **Table 3:** `rq3a_hallucination_rates.csv` - Per-domain hallucination statistics
4. **Table 4:** `rq3c_feature_importance_variability.csv` - Feature CV analysis
5. **Table 5:** Individual `metrics_*.json` files (this directory) - Full metrics per domain

### Statistical Test Results

**Location:** [`../research_questions/statistical_tests_summary.json`](../research_questions/statistical_tests_summary.json)

Contains:
- RQ1: t-statistic, p-value, Cohen's d
- RQ2: t-statistic, p-value, Cohen's d  
- RQ3a: Chi-square statistic, p-value
- RQ3c: Feature variability metrics

---

## Methodology Summary

**Approach:** Per-domain ablation with cross-domain statistical comparison

**Key Features:**
- ✅ No cross-domain data leakage (each domain analyzed independently)
- ✅ Identical XGBoost configuration for fair feature subset comparisons
- ✅ Fixed train/test splits (80/20 stratified) per domain
- ✅ Proper statistical tests (paired t-tests, chi-square)
- ✅ Effect sizes reported (Cohen's d)
- ✅ Reproducible (random_state=42 throughout)

**Total Models Trained:** 30
- 25 ablation models (5 domains × 5 feature subsets)
- 5 full per-domain models (with hyperparameter tuning)

**Execution Date:** February 14, 2026

---

## Archive Contents

The `archive/` subdirectory contains files from earlier research iterations:
- Old 3-domain comparisons (before IS Agents and complete Finance runs)
- Intermediate validation reports
- Calibration experiments
- Development summaries from Feb 7-13

These are preserved for historical reference but superseded by current analysis.

---

## Contact and Version

**Project:** HHPF (Hybrid Hallucination Prediction Framework)  
**Research Type:** Master's Thesis  
**Completion Date:** February 14, 2026  
**Status:** ✅ Thesis-Ready

**For Questions:** Refer to main project documentation in [`docs/`](../../docs/)
