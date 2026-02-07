# HHPF Documentation Index

**Last Updated:** 2026-02-07 13:30  
**Domains Complete:** 4/5 (Math ✅, Finance ✅, Medicine ✅, Psychology ✅)  
**Research Questions:** ✅ All 3 answered (RQ1, RQ2, RQ3)

---

## Quick Navigation

### 📋 **Start Here**
- **[START_HERE.md](../START_HERE.md)** - Project overview and getting started guide
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status and latest updates
- **[README.md](../README.md)** - Repository README

### 🎯 **Research**
- **[RESEARCH_QUESTIONS.md](../RESEARCH_QUESTIONS.md)** - Research questions and hypotheses
- **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** - Complete run history (RUN-001 through RUN-016)
- **[outputs/research_questions/](../outputs/research_questions/)** - ✨ **NEW:** RQ1, RQ2, RQ3 analysis outputs
- **[RESEARCH_ACTION_PLAN.md](RESEARCH_ACTION_PLAN.md)** - ✨ **NEW:** Plan to complete full research

---

## Domain Documentation

### ✅ Math Domain (GSM8K) - COMPLETE

**Status:** Production-ready | AUROC: 0.7918 | Samples: 542

**Key Documents:**
- **[outputs/results/domain1_math_summary.md](../outputs/results/domain1_math_summary.md)** - Complete results
- **[SEMANTIC_ENTROPY_FIX_SUMMARY.md](SEMANTIC_ENTROPY_FIX_SUMMARY.md)** - Critical bug fix details
- **[outputs/figures/math/](../outputs/figures/math/)** - All visualizations (5 sets)

**Metrics:** [outputs/results/metrics_math.json](../outputs/results/metrics_math.json)

**Key Achievement:** Fixed semantic entropy NULL values (500/500 → 0/542), improved AUROC from 0.50 to 0.79

---

### ✅ Finance Domain (FinanceBench) - COMPLETE

**Status:** Production-ready | AUROC: 0.6827 | Samples: 150

**Key Documents:**
- **[FINANCE_LABELING_FIX.md](FINANCE_LABELING_FIX.md)** - Labeling bug fix details
- **[outputs/results/domain2_finance_summary.md](../outputs/results/domain2_finance_summary.md)** - Results analysis
- **[outputs/figures/finance/](../outputs/figures/finance/)** - All visualizations (5 sets)
- **[outputs/figures/finance/README.md](../outputs/figures/finance/README.md)** - Visualization guide

**Metrics:** [outputs/results/metrics_finance.json](../outputs/results/metrics_finance.json)

**Key Achievement:** Fixed unit normalization bug (100% → 86% hallucination rate), validated verification process

---

### ✅ Medicine Domain (Med-HALT) - COMPLETE ✨

**Status:** Production-ready | AUROC: 0.6007 | Samples: 500 (reprocessed)

**Key Documents:**
- **[MEDICINE_DOMAIN_FIX.md](MEDICINE_DOMAIN_FIX.md)** - Consolidated fix documentation ⭐
- **[MEDICINE_DOMAIN_RESULTS.md](MEDICINE_DOMAIN_RESULTS.md)** - ✨ **NEW:** Final results and interpretation
- **[outputs/figures/medicine/](../outputs/figures/medicine/)** - All visualizations (5 sets)

**Metrics:** [outputs/results/metrics_medicine.json](../outputs/results/metrics_medicine.json)

**Key Challenges Resolved:**
- NULL ground truth handling (1,860 samples filtered)
- "None of the above" answer type requires special semantic handling
- Stratified splitting bug fixed (0% train/test gap achieved)
- Threshold tuning: 0.50 combined score optimal

**Key Achievement:** Successfully reprocessed with stratification fix, 0% train/test gap

---

### ✅ Psychology Domain (TruthfulQA) - COMPLETE ✨

**Status:** Production-ready | AUROC: 0.7115 | Samples: 500 (reprocessed)

**Key Documents:**
- **[PSYCHOLOGY_LABELING_FIX.md](PSYCHOLOGY_LABELING_FIX.md)** - Semantic similarity implementation ⭐
- **[outputs/figures/psychology/](../outputs/figures/psychology/)** - All visualizations (5 sets)

**Metrics:** [outputs/results/metrics_psychology.json](../outputs/results/metrics_psychology.json)

**Key Innovation:**
- **Problem:** Text similarity completely fails (2-3% scores even for correct answers)
- **Solution:** Semantic similarity using sentence embeddings (SentenceTransformer)
- **Model:** all-MiniLM-L6-v2 with 0.7 cosine similarity threshold
- **Impact:** Captures semantic meaning rather than character overlap

**Key Achievement:** Successfully reprocessed with stratification fix, 0.5% train/test gap

---

### ⏳ IS/Agents Domain (HalluMix) - PENDING

**Status:** Dataset ready (37,727 rows) | Code ready | Awaiting user run

**Dataset:** `../data/raw/hallumix.csv`  
**Has Labels:** Yes (pre-existing `hallucination_label` column)  
**Fix Applied:** `existing_label` handling in `src/data_preparation/process_datasets.py` and `src/data_preparation/label_responses.py`

**Status:** Ready for 100-sample test, then 500-sample run

---

## Research Questions Analysis ✨ NEW

### Overview
**Location:** [outputs/research_questions/](outputs/research_questions/)  
**Domains Analyzed:** 4 (Math, Finance, Medicine, Psychology)  
**Total Samples:** 1,692

### RQ1: Feature Hypothesis ✅ ANSWERED

**Question:** Do hybrid features (semantic + contextual) outperform baselines?

**Result:** ✓ SUPPORTED

**Key Findings:**
- **Best Model:** Semantic + Context (AUROC 0.8109)
- **Full Model:** 0.7936 (slightly lower, possible overfitting)
- **Semantic Only:** 0.7788
- **Context Only:** 0.7613
- **Naive Baseline:** 0.5000 (random - logprobs unavailable)

**Files:**
- [rq1_ablation_study.csv](../outputs/research_questions/rq1_ablation_study.csv)
- [figures/rq1_ablation_comparison.png|pdf](../outputs/research_questions/figures/)

---

### RQ2: Semantic vs Naive ⚠️ LIMITED (see LOGPROBS_LIMITATION.md)

**Question:** Does Semantic Entropy outperform naive confidence metrics?

**Result:** Technically SUPPORTED, but with caveat

**Key Findings:**
- **Semantic AUROC:** 0.7788
- **Naive AUROC:** 0.5000 (random - logprobs unavailable)
- **Improvement:** +0.2788 (+55.8%)
- **Interpretation:** Semantic entropy works **without logprobs**, making it practical for any API

**Files:**
- [rq2_semantic_vs_naive.csv](../outputs/research_questions/rq2_semantic_vs_naive.csv)
- [figures/rq2_semantic_vs_naive.png|pdf](../outputs/research_questions/figures/)

---

### RQ3: Cross-Domain Variance ✅ ANSWERED

**Question:** Do hallucination signatures differ significantly across domains?

**Result:** ✓ STRONGLY SUPPORTED

**Key Findings:**
- **Chi-square test:** χ² = 556.22, p < 0.001 (highly significant)
- **Domain-specific AUROCs:**
  - Math: 0.6855
  - Finance: 0.5865
  - Medicine: 0.5751
  - Psychology: 0.4276
- **Hallucination rates vary:** 29% (Math) to 91% (Medicine)
- **21 features** show high cross-domain variation (CV > 0.3)
- **Top varying features:** entity_type_EVENT, qtype_why, num_rare_entities, avg_cluster_size

**Files:**
- [rq3_domain_metrics.csv](../outputs/research_questions/rq3_domain_metrics.csv)
- [rq3_feature_importance_differences.csv](../outputs/research_questions/rq3_feature_importance_differences.csv)
- [figures/rq3_domain_auroc.png|pdf](../outputs/research_questions/figures/)
- [figures/rq3_domain_feature_heatmap.png|pdf](../outputs/research_questions/figures/)
- [domain_models/](../outputs/research_questions/domain_models/) - 4 domain-specific XGBoost models

---

## Bug Fixes & Technical Documentation

### Critical Bugs Fixed

1. **Semantic Entropy Bug (Math Domain)**
   - **Document:** [SEMANTIC_ENTROPY_FIX_SUMMARY.md](SEMANTIC_ENTROPY_FIX_SUMMARY.md)
   - **Issue:** All semantic entropy features NULL due to model reloading
   - **Impact:** AUROC 0.50 → 0.79 after fix
   - **Date:** 2026-02-05

2. **Finance Labeling Bug (Finance Domain)**
   - **Document:** [FINANCE_LABELING_FIX.md](FINANCE_LABELING_FIX.md)
   - **Issue:** 100% hallucination rate due to unit normalization
   - **Impact:** Training failure → Success with AUROC 0.68
   - **Date:** 2026-02-05

3. **Psychology Text Similarity Failure (Psychology Domain)**
   - **Document:** [PSYCHOLOGY_LABELING_FIX.md](PSYCHOLOGY_LABELING_FIX.md)
   - **Issue:** Text similarity (SequenceMatcher) completely fails for TruthfulQA
   - **Root Cause:** Short ground truths (5-15 words) vs long LLM responses (50-200 words) → 2-3% similarity scores
   - **Solution:** Semantic similarity using sentence embeddings (SentenceTransformer)
   - **Model:** all-MiniLM-L6-v2 with 0.7 threshold
   - **Impact:** Enables semantic equivalence detection despite different wording
   - **Date:** 2026-02-06
   - **Status:** ✅ Complete

4. **Stratification Bug (Medicine & Psychology)** ✨ **CRITICAL**
   - **Document:** [STRATIFICATION_BUG_AND_FIX.md](STRATIFICATION_BUG_AND_FIX.md) ⭐
   - **Issue:** Train/test split was either not stratified or stratified on wrong variable
   - **Root Cause:** `train_test_split` called on unlabeled data in `process_datasets.py`
   - **Impact:** Psychology AUROC 0.53 → 0.71 after fix; Medicine train/test gap eliminated
   - **Solution:** Restructured pipeline - label ALL responses BEFORE splitting
   - **New Module:** `src/data_preparation/label_responses.py`
   - **Date:** 2026-02-06
   - **Status:** ✅ Permanent fix implemented

### Limitations

- **[LOGPROBS_LIMITATION.md](LOGPROBS_LIMITATION.md)** - ✨ **NEW:** Comprehensive documentation of missing logprobs features and impact on RQ2

### Process Documentation

(Historical documents - may not exist in current version)

---

## Configuration Files

### Data Configuration
- **[configs/datasets.yaml](../configs/datasets.yaml)** - Dataset paths and parameters
- **[configs/features.yaml](../configs/features.yaml)** - Feature extraction settings
- **[configs/model.yaml](../configs/model.yaml)** - Model hyperparameters

### Project Setup
- **[requirements.txt](../requirements.txt)** - Python dependencies
- **[setup.py](../setup.py)** - Package installation
- **[.env.example](../.env.example)** - Environment variables template

---

## Source Code Documentation

### Pipeline Modules

**Data Preparation:**
- `../src/data_preparation/dataset_loaders.py` - Load domain datasets
- `../src/data_preparation/process_datasets.py` - Process and split data
- `../src/data_preparation/ground_truth.py` - Domain-specific labelers ⭐
- `../src/data_preparation/prompt_formatter.py` - Format prompts for LLMs

**Inference:**
- `../src/inference/llama_client.py` - LLM API clients (Groq, Together)
- `../src/inference/response_generator.py` - Generate stochastic samples

**Features:**
- `../src/features/epistemic_uncertainty.py` - Semantic entropy/energy ⭐
- `../src/features/contextual_features.py` - NER, question types, linguistic
- `../src/features/feature_aggregator.py` - Orchestrate feature extraction ⭐

**Classification:**
- `../src/classifier/xgboost_model.py` - XGBoost classifier
- `../src/classifier/hyperparameter_tuning.py` - HPO with Optuna

**Evaluation:**
- `../src/evaluation/metrics.py` - AUROC, ARC, ECE, calibration
- `../src/evaluation/visualization.py` - Generate plots
- `../src/evaluation/hypothesis_testing.py` - Statistical tests

**Pipeline:**
- `../run_pipeline.py` - End-to-end pipeline orchestration ⭐

---

## Experiment Logs

### Run History

| Run ID | Date | Domain | Samples | AUROC | Status | Notes |
|--------|------|--------|---------|-------|--------|-------|
| RUN-003 | 2026-02-05 | Math | 542 | 0.79 | ✅ Complete | Semantic entropy fix |
| RUN-004 | 2026-02-05 | Finance | 150 | 0.68 | ✅ Complete | Unit normalization fix |
| RUN-014 | 2026-02-06 | Psychology | 500 | 0.71 | ✅ Complete | Reprocessed with stratification fix |
| RUN-015 | 2026-02-07 | Medicine | 500 | 0.60 | ✅ Complete | Reprocessed with stratification fix |
| RUN-016 | 2026-02-07 | Combined | 1,692 | 0.81 | ✅ Complete | Research questions analysis (RQ1-RQ3) |

**Detailed Logs:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)  
**See also:** [STRATIFICATION_BUG_AND_FIX.md](STRATIFICATION_BUG_AND_FIX.md) for RUN-014 and RUN-015 context

---

## Outputs & Results

### Models
```
outputs/models/
├── xgboost_math.pkl        # Math domain classifier (3.7 MB)
└── xgboost_finance.pkl     # Finance domain classifier (177 KB)
```

### Metrics
```
outputs/results/
├── metrics_math.json                   # Math evaluation metrics
├── metrics_finance.json                # Finance evaluation metrics
├── domain1_math_summary.md             # Math analysis
└── domain2_finance_summary.md          # Finance analysis
```

### Visualizations
```
outputs/figures/
├── math/                               # Math domain plots (10 files)
│   ├── roc_curve_math.png|pdf
│   ├── confusion_matrix_math.png|pdf
│   ├── calibration_math.png|pdf
│   ├── arc_math.png|pdf
│   └── feature_importance_math.png|pdf
│
└── finance/                            # Finance domain plots (10 files)
    ├── roc_curve_finance.png|pdf
    ├── confusion_matrix_finance.png|pdf
    ├── calibration_finance.png|pdf
    ├── arc_finance.png|pdf
    ├── feature_importance_finance.png|pdf
    └── README.md                       # Visualization guide
```

---

## Data Files

### Raw Data
```
data/raw/
├── gsm8k.csv                           # Math word problems (8,792)
├── financebench_sample_150.csv         # Finance Q&A (150)
├── med_halt.csv                        # Medical questions (placeholder)
└── README.md
```

### Processed Data
```
data/processed/
├── math_processed.csv                  # 542 samples, train/test split
└── finance_processed.csv               # 150 samples, train/test split
```

### Features
```
data/features/
├── math_features.csv                   # 542×47 feature matrix
├── finance_features.csv                # 150×47 feature matrix
├── math_*_responses.pkl                # 542 response caches
└── finance_*_responses.pkl             # 150 response caches
```

---

## Scripts & Utilities

### Analysis Scripts
```
scripts/
├── prepare_datasets.py                 # Prepare all domain datasets
├── inspect_datasets.py                 # Quick dataset inspection
└── analyze_research_questions.py      # RQ analysis tools
```

### Notebooks
```
notebooks/
├── 00_setup_verification.ipynb         # Environment setup check
└── 01_data_exploration.ipynb           # Dataset exploration
```

---

## Key Findings Summary

### Math Domain (GSM8K)
- **AUROC:** 0.7918 (excellent)
- **Key Finding:** Semantic entropy is top predictor after fix
- **Hallucination Rate:** ~50% (balanced dataset)
- **Challenge:** Initial bug with model loading caused 8-hour runtime

### Finance Domain (FinanceBench)
- **AUROC:** 0.6827 (good, despite challenges)
- **Key Finding:** Finance significantly harder than math (86% vs 50% hallucinations)
- **Challenge:** Unit normalization required for accurate labeling
- **Success:** Verification process caught critical bug early

### Cross-Domain Insights
1. **Epistemic uncertainty works across domains** (0% NULL in both)
2. **Domain difficulty varies significantly** (50% vs 86% hallucination)
3. **Domain-specific labeling critical** (generic approaches fail)
4. **Systematic verification essential** (caught 2 major bugs)

---

## For New Domains

### Checklist Before Running
1. Read [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md)
2. Check dataset in `data/raw/`
3. Run 10-sample test first
4. Verify label distribution (both classes?)
5. Check semantic entropy (0% NULL?)
6. Review labeling logic for domain-specific needs

### Expected Files After Run
- `data/processed/{domain}_processed.csv`
- `data/features/{domain}_features.csv`
- `data/features/{domain}_*_responses.pkl` (one per sample)
- `outputs/models/xgboost_{domain}.pkl`
- `outputs/results/metrics_{domain}.json`
- `outputs/figures/{domain}/*.png|pdf` (5 visualization sets)

---

## Troubleshooting

### Common Issues

1. **Semantic entropy features NULL**
   - Check: `grep -n "self.entropy_calc = None" src/features/feature_aggregator.py`
   - Should exist in FeatureAggregator class
   - See: [SEMANTIC_ENTROPY_FIX_SUMMARY.md](SEMANTIC_ENTROPY_FIX_SUMMARY.md)

2. **100% of one label**
   - Check labeling logic in `src/data_preparation/ground_truth.py`
   - May need domain-specific adjustments
   - See: [FINANCE_LABELING_FIX.md](FINANCE_LABELING_FIX.md)

3. **Training fails with "Invalid classes"**
   - Cause: Only one class in dataset
   - Fix: Review labeling logic, check ground truth
   - Verify: Both classes present in `data/features/{domain}_features.csv`

4. **AUROC ~0.50 (random)**
   - Check for NULL features (esp. semantic entropy)
   - Verify model loaded correctly
   - Review feature extraction logs

---

## Citation & References

### This Project
```bibtex
@mastersthesis{gross2026hhpf,
  author = {Gross, Aviv},
  title = {Hybrid Hallucination Prediction Framework: 
           Combining Epistemic Uncertainty with Contextual Features},
  school = {[University Name]},
  year = {2026},
  note = {In progress}
}
```

### Key Papers Referenced
- Semantic Entropy: Kuhn et al. (2023)
- Hallucination Detection: Lin et al. (2021)
- GSM8K Dataset: Cobbe et al. (2021)
- FinanceBench: [Citation needed]

---

## Changelog

### 2026-02-05 (Finance Domain Complete)
- ✅ Completed finance domain pipeline
- ✅ Fixed unit normalization bug in FinanceLabeler
- ✅ Generated all finance visualizations
- ✅ Created comprehensive documentation
- ✅ Updated EXPERIMENT_LOG with RUN-004
- ✅ Validated verification process

### 2026-02-05 (Math Domain Complete)
- ✅ Fixed semantic entropy NULL bug
- ✅ Completed math domain with AUROC 0.79
- ✅ Generated all visualizations
- ✅ Created semantic entropy fix summary

### 2026-02-04 (Initial Runs)
- ⚠️ RUN-001: Initial 50-sample test
- ⚠️ RUN-002: Discovered semantic entropy bug

---

## Next Steps

### Completed ✅
- [x] Document finance domain
- [x] Medicine domain (Med-HALT)
- [x] Psychology domain (TruthfulQA)
- [x] Cross-domain comparison analysis (RQ1, RQ2, RQ3)
- [x] Stratification bug fix and reprocessing
- [x] Comprehensive documentation consolidation

### Pending
- [ ] IS/Agents domain (HalluMix) - Dataset ready, awaiting user run
- [ ] Thesis writing with research questions results
- [ ] (Optional) Implement logprobs support for comprehensive RQ2 validation

---

## Contact & Support

**Project Author:** Aviv Gross  
**Institution:** [University Name]  
**Degree:** Master's Thesis  
**Year:** 2026

**For Questions:**
- See [START_HERE.md](../START_HERE.md) for setup instructions
- Check [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for run history
- Check [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for comprehensive documentation index

---

**Last Updated:** 2026-02-05 09:30  
**Status:** 2/5 domains complete (Math ✅, Finance ✅)  
**Next:** Medicine domain verification and execution
