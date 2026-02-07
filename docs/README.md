# HHPF Documentation

This directory contains all technical documentation, bug reports, domain analyses, and research artifacts for the Hybrid Hallucination Prediction Framework (HHPF).

---

## Quick Start

**New to the project?** Start with the root-level files:
- **[../README.md](../README.md)** - Project overview and setup
- **[../START_HERE.md](../START_HERE.md)** - Getting started guide
- **[../RESEARCH_QUESTIONS.md](../RESEARCH_QUESTIONS.md)** - Research questions and hypotheses

---

## Documentation Structure

### 📊 Central Index
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Comprehensive project documentation index

### 🔬 Research & Planning
- **[RESEARCH_ACTION_PLAN.md](RESEARCH_ACTION_PLAN.md)** - Plan to complete remaining research (IS/Agents domain)
- **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** - Complete experiment history (RUN-001 through RUN-016)

### 🐛 Bug Reports & Fixes
- **[STRATIFICATION_BUG_AND_FIX.md](STRATIFICATION_BUG_AND_FIX.md)** ⭐ - **CRITICAL**: Train/test split stratification bug
- **[SEMANTIC_ENTROPY_FIX_SUMMARY.md](SEMANTIC_ENTROPY_FIX_SUMMARY.md)** - Math domain semantic entropy NULL bug
- **[CRITICAL_BUG_FOUND.md](CRITICAL_BUG_FOUND.md)** - Original stratification bug discovery

### 📈 Domain-Specific Documentation

#### Math Domain (GSM8K) - AUROC 0.79
- **[SEMANTIC_ENTROPY_FIX_SUMMARY.md](SEMANTIC_ENTROPY_FIX_SUMMARY.md)** - NULL semantic entropy fix

#### Finance Domain (FinanceBench) - AUROC 0.68
- **[FINANCE_LABELING_FIX.md](FINANCE_LABELING_FIX.md)** - Unit normalization bug fix

#### Medicine Domain (Med-HALT) - AUROC 0.60
- **[MEDICINE_DOMAIN_FIX.md](MEDICINE_DOMAIN_FIX.md)** ⭐ - Consolidated medicine fixes
- **[MEDICINE_DOMAIN_RESULTS.md](MEDICINE_DOMAIN_RESULTS.md)** - Final results interpretation

#### Psychology Domain (TruthfulQA) - AUROC 0.71
- **[PSYCHOLOGY_LABELING_FIX.md](PSYCHOLOGY_LABELING_FIX.md)** - Semantic similarity implementation

### ⚠️ Limitations & Interpretations
- **[LOGPROBS_LIMITATION.md](LOGPROBS_LIMITATION.md)** ⭐ - Missing logprobs analysis and RQ2 interpretation
- **[OVERNIGHT_RESULTS_ANALYSIS.md](OVERNIGHT_RESULTS_ANALYSIS.md)** - Initial 500-sample runs analysis

### 📝 Project Status
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Historical project status snapshots

---

## Key Insights Summary

### Completed Work (4/5 Domains)
- ✅ **Math:** 0.79 AUROC (542 samples)
- ✅ **Finance:** 0.68 AUROC (150 samples)  
- ✅ **Medicine:** 0.60 AUROC (500 samples)
- ✅ **Psychology:** 0.71 AUROC (500 samples)
- ⏳ **IS/Agents:** Pending user run

### Research Questions Answered
- **RQ1 (Feature Hypothesis):** ✓ SUPPORTED - Semantic+Context best (0.81 AUROC)
- **RQ2 (Semantic vs Naive):** ⚠️ LIMITED - Logprobs unavailable, but semantic entropy proven effective
- **RQ3 (Cross-Domain Variance):** ✓ STRONGLY SUPPORTED - χ²=556, p<0.001

### Critical Bugs Fixed
1. **Stratification Bug** - Train/test split on unlabeled data → Psychology 0.53→0.71, Medicine gap eliminated
2. **Semantic Entropy NULL** - Model reloading issue → Math 0.50→0.79
3. **Finance Unit Normalization** - 100% hallucinations → 86% correct rate
4. **Psychology Text Similarity** - Semantic similarity implementation with SentenceTransformer

---

## Usage

All documentation files use relative links. You can:
- Browse on GitHub
- Read locally in any markdown viewer
- Use `DOCUMENTATION_INDEX.md` as the main navigation hub

---

## For Thesis Writing

The most important documents for thesis citations:
1. **[STRATIFICATION_BUG_AND_FIX.md](STRATIFICATION_BUG_AND_FIX.md)** - Methodology improvement
2. **[LOGPROBS_LIMITATION.md](LOGPROBS_LIMITATION.md)** - Limitations section
3. **[MEDICINE_DOMAIN_FIX.md](MEDICINE_DOMAIN_FIX.md)** - Domain-specific challenges
4. **[PSYCHOLOGY_LABELING_FIX.md](PSYCHOLOGY_LABELING_FIX.md)** - Technical innovation

---

**Last Updated:** 2026-02-07  
**Status:** 4/5 domains complete, thesis-ready
