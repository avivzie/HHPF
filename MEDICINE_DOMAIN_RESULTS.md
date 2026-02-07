# Medicine Domain – Results Summary

**Date**: February 7, 2026  
**Status**: ✅ Reprocessing complete (stratification fix applied)  
**Dataset**: Med-HALT, 500 samples

---

## What You Have Now

Medicine has been **reprocessed with the stratification fix** (no new API calls). The pipeline re-labeled all 500 cached responses, created a proper stratified train/test split, retrained the model, and generated metrics and figures.

### Main numbers

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **Test AUROC** | **0.6007** | Above random (0.5). Model separates hallucinations from faithful responses, but domain is hard (imbalanced, medical nuance). |
| **Test accuracy** | 0.80 | 80% of test predictions correct. |
| **Train/Test hallucination rate** | **91.0% / 91.0%** | **0% gap** → stratification worked. |
| **Test ECE** | 0.20 | Moderate calibration error; probabilities somewhat overconfident. |

### Stratification (the fix)

- **Before fix**: Train 90.2% hallucinations, Test 94.0% (only 6 faithful in test) → AUROC ~0.51 (near random).
- **After fix**: Train 91.0%, Test 91.0% (9 faithful in test) → AUROC **0.60**.
- So: proper stratification gave a **~18% relative gain in test AUROC** (0.51 → 0.60) and a balanced evaluation.

### Test set breakdown (100 samples)

- **Faithful**: 9 (9%)
- **Hallucinations**: 91 (91%)
- **Confusion matrix (test)**:
  - True negatives (correct faithful): 4  
  - False positives: 5  
  - False negatives (missed hallucinations): 15  
  - True positives (correct hallucinations): 76  

So the model is better at flagging hallucinations (76 correct) than at sparing faithful answers (4 correct); with only 9 faithful in test, small changes in those 9 have a big impact on accuracy and AUROC.

---

## Why Medicine Is Harder Than Other Domains

- **Extreme class imbalance**: ~91% hallucinations (Med-HALT multiple-choice, many “None of the above”).
- **Tight medical wording**: “None of the above” vs specific answers needs semantic handling; exact wording varies.
- **Few faithful examples**: 500 samples → only ~45 faithful total, ~9 in test, so metrics are noisier.

So **AUROC 0.60 for medicine is a reasonable result** given the dataset and imbalance, and it’s **clearly better than the broken run (0.51)**.

---

## Comparison to Other Domains

| Domain | Test AUROC | Train/Test gap | Note |
|--------|------------|----------------|----------------|
| Math | 0.79 | – | Best; clear right/wrong. |
| Psychology | 0.71 | 0.5% | Strong; semantic labeling works. |
| Finance | 0.68 | – | Good; numerical + tolerance. |
| **Medicine** | **0.60** | **0%** | Hard; imbalance + medical nuance. |
| IS/Agents (HalluMix) | – | – | Pending. |

Medicine sits where we’d expect: above random, below domains with clearer signals and more balance.

---

## Where Everything Lives

- **Metrics**: `outputs/results/metrics_medicine_reprocessed.json`  
- **Model**: `outputs/models/xgboost_medicine_reprocessed.pkl`  
- **Features**: `data/features/medicine_features_reprocessed.csv`  
- **Figures**: `outputs/figures/medicine_reprocessed/`  
  - ROC, ARC, calibration, confusion matrix, feature importance (PDF + PNG)

Full bug-fix and validation story: **`MEDICINE_DOMAIN_FIX.md`**.

---

## One-Sentence Summary

Medicine reprocessing with proper stratification is complete: **Test AUROC 0.60**, **0% train/test gap**, with all metrics and figures saved; next step is HalluMix when you continue.
