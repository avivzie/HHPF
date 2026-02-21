# Feedback-Driven Changes and Draft Update Guide

**Date:** February 21, 2026  
**Purpose:** Single reference for all changes made in response to feedback (recent commits), the results/figures that changed, and actionable guidance for updating your thesis/paper draft with the fixed versions.

---

## Overview Table

| Feedback | Resolution | Results/figures changed | Detail |
|----------|------------|------------------------|--------|
| **Cross-validation** | 5-fold stratified CV added to all model evaluations | Ablation CSVs and RQ summaries now include CV columns (mean ± std); new single-split vs CV comparison file | [INSTRUCTOR_FEEDBACK_AUDIT.md](INSTRUCTOR_FEEDBACK_AUDIT.md) (Feedback 1) |
| **Statistical rigor** | Confidence intervals, Bonferroni and FDR corrections for multiple comparisons | Statistical test outputs and summaries updated with 95% CIs, raw and corrected p-values | [INSTRUCTOR_FEEDBACK_AUDIT.md](INSTRUCTOR_FEEDBACK_AUDIT.md) (Feedback 2) |
| **AUROC consistency** | Single pipeline for training and figures; RQ3b and ROC curves use same metrics source | All domain AUROCs consistent across outputs; figures from `metrics_{domain}.json` | [outputs/results/CONSISTENCY_IMPLEMENTATION_SUMMARY.md](outputs/results/CONSISTENCY_IMPLEMENTATION_SUMMARY.md) |
| **Overinterpretation of AUROC / imbalanced domains** | PR-AUC and F1 curves added; optimal F1 threshold reported | New PR curve figures per domain (two-panel: PR curve + F1 vs threshold) | Section below (Feedback 3) |

---

## 1. PR-AUC and F1 Curves (Feedback 3)

### Feedback

"Overinterpretation of AUROC alone – AUROC is presented as a key metric, but in a subject with an imbalance in influence (such as psychology and mathematics) PR-AUC was more informative. This can be resolved by adding Precision-Recall and F1 curves according to an optimal connection."

### What we did

- **`src/evaluation/metrics.py`:** Added `calculate_pr_curve()` using sklearn’s `precision_recall_curve` and `average_precision_score`; AUPRC and optimal F1 threshold (and curve data) are now included in `calculate_all_metrics()` under `auprc` and `pr_curve`.
- **`scripts/generate_domain_figures.py`:** Added `plot_pr_curve()` producing a two-panel figure:
  - **Left:** Precision-Recall curve with AUPRC in legend, random baseline (positive rate), and optimal F1 point marked.
  - **Right:** F1 score vs decision threshold with optimal threshold marked.

Existing metrics JSONs (without `pr_curve`) are supported: PR data is derived from ROC curve + confusion matrix when needed.

### Results/figures changed

- **New outputs** (per domain):  
  `outputs/figures/{domain}/pr_curve_{domain}.pdf` and `pr_curve_{domain}.png`  
  for **math**, **is_agents**, **psychology**, **medicine**, **finance**.
- Future runs of `train_consistent_models.py` will write `auprc` and `pr_curve` into `outputs/results/metrics_{domain}.json`.

### For your draft

- **Methods:** Add one sentence that for imbalanced domains we report AUPRC (area under the precision-recall curve) and the decision threshold that maximizes F1, alongside AUROC.
- **Results/Figures:** Add or reference the PR curve figure(s), e.g. *"Figure X: Precision-Recall curve and F1 vs decision threshold for [domain(s)]."* Consider at least **psychology** and **math** as examples; all five domains have figures available.
- **Text:** State that AUROC remains the primary discrimination metric, but AUPRC and F1 at the optimal threshold are reported for imbalanced domains (e.g. psychology, mathematics) where they are more informative.

---

## 2. Cross-Validation and Statistical Rigor (Summary)

### What changed

- **Cross-validation:** All ablation and full models are evaluated with 5-fold stratified cross-validation on the full dataset; results are reported as mean ± std alongside single-split metrics.
- **Statistical rigor:** 95% confidence intervals (t-based, n=5 domains), Bonferroni and FDR corrections for the three primary tests (RQ1, RQ2, RQ3a), and effect sizes (e.g. Cohen’s d) are now reported.

### Results that changed

- RQ1/RQ2 **point estimates** are unchanged (e.g. +0.116 AUROC for RQ1).
- **Now also reported:** 95% CIs for mean differences, raw and corrected p-values (Bonferroni, FDR), CV means ± std.
- **New/updated files:**  
  `outputs/research_questions/statistical_tests_summary_cv.json`,  
  `outputs/research_questions/multiple_comparison_corrections.csv`,  
  `outputs/research_questions/statistical_tests_comparison.json`,  
  and enhanced `statistical_tests_summary_single_split.json`.

### For your draft

- Use **"Key Results Changes"** and **"Example reporting format"** in [INSTRUCTOR_FEEDBACK_AUDIT.md](INSTRUCTOR_FEEDBACK_AUDIT.md) to replace any narrative that only mentioned "borderline significance."
- Report for RQ1/RQ2: **point estimate**, **95% CI**, **raw p-value**, **corrected p-value(s)**, and **effect size**.

**Example reporting format** (from INSTRUCTOR_FEEDBACK_AUDIT):

```
Hybrid features showed a mean improvement of +0.116 AUROC (95% CI: [-0.027, 0.259]) 
over naive baselines, with a large effect size (Cohen's d = 1.007). While the raw 
p-value of 0.087 suggested borderline significance, this did not survive Bonferroni 
correction for multiple comparisons (p_corrected = 0.262). However, the large 
effect size suggests practical importance warranting further investigation.
```

---

## 3. AUROC Consistency (Summary)

### What changed

The thesis had different AUROC values in the RQ3b domain chart vs the individual ROC curves (differences up to ~0.06). A single training and figure-generation pipeline was introduced so that RQ3b and all ROC (and other) figures use the same metrics source (`metrics_{domain}.json` from `scripts/train_consistent_models.py` and `scripts/generate_domain_figures.py`).

### Results that changed

- All reported domain AUROCs are now consistent across RQ3b, ROC curves, and summaries.
- **Final consistent AUROC values** (use these in the draft):

| Domain     | AUROC (Consistent) |
|-----------|---------------------|
| Math      | 0.7973              |
| IS Agents | 0.7027              |
| Psychology| 0.6715              |
| Medicine  | 0.6192              |
| Finance   | 0.6320              |

**Mean AUROC:** 0.6845 ± 0.0711

### For your draft

- Ensure every reported AUROC and every figure caption uses these values (or the rounded versions 0.797, 0.703, 0.671, 0.619, 0.632).
- Confirm that the RQ3b domain chart and the individual ROC curve figures all cite the same numbers; source: [outputs/results/5_DOMAIN_FINAL_SUMMARY.md](outputs/results/5_DOMAIN_FINAL_SUMMARY.md) or [outputs/results/CONSISTENCY_IMPLEMENTATION_SUMMARY.md](outputs/results/CONSISTENCY_IMPLEMENTATION_SUMMARY.md).

---

## 4. Checklist for Updating the Draft

- [ ] **Methods:** Add a sentence that model evaluation includes 5-fold stratified cross-validation and where CV results are reported (e.g. ablation tables, RQ summaries).
- [ ] **Methods:** Add a sentence on reporting 95% confidence intervals and multiple comparison corrections (Bonferroni, FDR) for the primary tests.
- [ ] **Methods:** Add a sentence that for imbalanced domains we report AUPRC and the optimal F1 threshold alongside AUROC.
- [ ] **Results:** Replace any single-split-only RQ1/RQ2 wording with the full reporting template (estimate, 95% CI, raw p, corrected p, effect size) from INSTRUCTOR_FEEDBACK_AUDIT.
- [ ] **Results:** Add PR curve figure(s) and at least one sentence on AUPRC/F1 for psychology and math (or all domains).
- [ ] **Figures:** Use the consistent AUROC source; confirm RQ3b and ROC curve numbers match the table above.
- [ ] **Figures:** Add PR curve figure paths or references for each domain (or for a representative subset, e.g. psychology and math).

---

## 5. File and Figure Reference

| Output | Path | Use in draft |
|--------|------|--------------|
| Final domain metrics and RQ results | `outputs/results/5_DOMAIN_FINAL_SUMMARY.md` | Main numbers, RQ1/RQ2/RQ3 tables, interpretation |
| PR curve figures (per domain) | `outputs/figures/{domain}/pr_curve_{domain}.pdf` (or `.png`) | Add as "Precision-Recall and F1 vs threshold" figure(s) |
| RQ statistical results | `outputs/research_questions/RESULTS_SUMMARY.md`, `outputs/research_questions/statistical_tests_summary_single_split.json`, `statistical_tests_summary_cv.json` | CIs, p-values, corrected p-values |
| Consistent AUROC table | `outputs/results/CONSISTENCY_IMPLEMENTATION_SUMMARY.md` (final table) | Copy-paste AUROC values for all narrative and figure captions |
| Full feedback implementation detail | `INSTRUCTOR_FEEDBACK_AUDIT.md` | Cross-validation and statistical rigor (Feedback 1 & 2) |

---

**Last updated:** February 21, 2026
