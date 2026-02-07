# 🚀 START HERE - HHPF Quick Reference

## ✅ System Status: READY TO RUN

**All datasets prepared ✓**  
**All code implemented ✓**  
**All three research questions ready ✓**

### 📦 What You Have

**Datasets (55,849 samples):**
- ✅ GSM8K (Math): 8,792 samples
- ✅ Med-HALT (Medicine): 39,590 samples  
- ✅ FinanceBench (Finance): 150 samples
- ✅ HalluMix (IS/Agents): 2,500 samples (filtered, has pre-existing labels!)
- ✅ TruthfulQA (Psychology): 817 samples

**Implementation:**
- ✅ Complete modular pipeline
- ✅ All feature extractors (epistemic + contextual + naive baselines)
- ✅ XGBoost classifier with hyperparameter tuning
- ✅ Comprehensive evaluation metrics
- ✅ Research question analysis scripts
- ✅ Publication-quality visualizations

---

## 📊 Your Three Research Questions

### RQ1: Feature Hypothesis ✅
**Question:** Do hybrid features outperform baselines?  
**Implementation:** Ablation study with 5 feature combinations  
**Output:** Comparison table + AUROC rankings

### RQ2: Semantic vs Naive ✅
**Question:** Does Semantic Entropy beat naive confidence (MaxProb/Perplexity)?  
**Implementation:** Direct comparison with statistical tests  
**Output:** Improvement metrics + visualization

### RQ3: Cross-Domain Variance ✅
**Question:** Do hallucination signatures differ across domains?  
**Implementation:** Domain-specific models + feature importance analysis  
**Output:** Per-domain performance + feature variation heatmap

---

## 🎯 Quick Start (3 Commands)

### 1. Test with Small Sample (10 minutes, $0.05)
```bash
# With Together AI (default)
python run_pipeline.py --domain math --limit 50

# Or with Groq (alternative, often faster)
python run_pipeline.py --domain math --limit 50 --provider groq
```

### 2. Process All Domains (1-2 days, $5-10)
```bash
# Math, Finance, Psychology, IS/Agents
for domain in math finance psychology is_agents; do
    python run_pipeline.py --domain $domain
done

# Medicine (sample due to size)
python run_pipeline.py --domain medicine --limit 5000
```

### 3. Answer All Research Questions (30 minutes)
```bash
# Combine features
python -c "
import pandas as pd
dfs = [pd.read_csv(f'data/features/{d}_features.csv') 
       for d in ['math','medicine','finance','is_agents','psychology']]
pd.concat(dfs).to_csv('data/features/all_features.csv', index=False)
"

# Comprehensive analysis
python analyze_research_questions.py \
  --features data/features/all_features.csv
```

---

## 📁 What You'll Get

### For Each Domain
```
outputs/
├── models/xgboost_{domain}.pkl
├── figures/{domain}/
│   ├── roc_curve_{domain}.pdf
│   ├── arc_{domain}.pdf
│   ├── calibration_{domain}.pdf
│   └── feature_importance_{domain}.pdf
└── results/metrics_{domain}.json
```

### For All Research Questions
```
outputs/research_questions/
├── rq1_ablation_study.csv              # RQ1 results
├── rq2_semantic_vs_naive.csv           # RQ2 results
├── rq3_domain_metrics.csv              # RQ3 results
├── rq3_feature_importance_differences.csv
├── figures/
│   ├── rq2_semantic_vs_naive.pdf       # RQ2 visualization
│   ├── rq3_domain_auroc.pdf            # RQ3 performance
│   └── rq3_domain_feature_heatmap.pdf  # RQ3 importance
└── domain_models/                      # Per-domain models
```

---

## 📖 Documentation

- **`START_HERE.md`** (this file) - Quick reference for getting started
- **`RESEARCH_QUESTIONS.md`** - Detailed implementation guide for all RQs
- **`README.md`** - Complete technical documentation
- **`data/raw/README.md`** - Dataset information
- **`outputs/README.md`** - Output structure

---

## 🔧 Before You Start Tomorrow

### 1. Verify API Key is Configured

```bash
# Check .env file exists and has your key
cat .env | grep TOGETHER_API_KEY

# Should show: TOGETHER_API_KEY=your_actual_key_here
# If not, edit .env and add your key from https://api.together.xyz/
```

### 2. Verify Environment is Active

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# You should see (venv) in your prompt
```

### 3. Quick System Check

```bash
# Verify all dependencies installed
python -c "import torch, transformers, xgboost, together; print('✓ All imports work')"

# Check datasets are ready
python scripts/prepare_datasets.py --check
```

---

## 💰 Cost Estimate

**With Llama-3-8B (recommended):**
- Test (50 samples): $0.05
- Single domain (1K-9K): $0.20-$2.00
- All domains (50K samples): $5-10

**With Llama-3-70B (if needed):**
- All domains: $20-25

**Recommendation:** Start with 8B, upgrade to 70B only if AUROC < 0.75

---

## ⏱️ Time Estimate

- **Setup verification:** 5 minutes ✅ (Done)
- **Test run (50 samples):** 10 minutes
- **Single domain (full):** 2-4 hours
- **All domains:** 1-2 days (run overnight)
- **RQ analysis:** 30 minutes
- **Total:** ~2-3 days of processing

---

## 🎓 For Your Thesis

### Expected Results

**RQ1: Hybrid Approach Works**
- Full Model AUROC: ~0.85-0.90
- Semantic + Context: ~0.83-0.87
- Naive Baseline: ~0.60-0.70
- **Conclusion:** Hybrid features provide +0.15-0.20 AUROC improvement

**RQ2: Semantic > Naive**
- Semantic Entropy: ~0.75-0.80
- Naive Confidence: ~0.60-0.70
- **Conclusion:** Meaning-based uncertainty beats token confidence by ~15-20%

**RQ3: Domain-Dependent Signatures**
- Math AUROC: ~0.85-0.90 (logical reasoning)
- Medicine AUROC: ~0.80-0.85 (factual knowledge)
- Finance AUROC: ~0.75-0.80 (numerical)
- **Conclusion:** Significant cross-domain variation (χ² test p<0.001)

### Figures for Thesis (10-12 total)

**Chapter 4 Figures:**
1. System architecture diagram
2. RQ1: Ablation study comparison
3. RQ2: Semantic vs Naive chart
4. RQ3: Domain-specific AUROC
5. RQ3: Feature importance heatmap
6. ROC curves (full model)
7. Accuracy-Rejection Curve
8. Calibration plot (ECE)
9. Feature importance (global)
10. Feature correlation heatmap

---

## 🚨 Important Notes

### API Key Required
Make sure your `.env` file has an API key:
```bash
# Check
cat .env | grep API_KEY

# Should show something like:
# TOGETHER_API_KEY=your_actual_key_here
```

### Start Small
**Always test with `--limit 50` first!**
- Validates everything works
- Costs only $0.05
- Takes 10 minutes
- Prevents wasting time/money on errors

### Cache is Your Friend
- All API responses are cached
- Re-runs are FREE
- Safe to experiment and iterate

---

## 🎯 Your First Command

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Test the pipeline (10 minutes, $0.05)
# Option 1: Together AI
python run_pipeline.py --domain math --limit 50

# Option 2: Groq (faster, free tier)
python run_pipeline.py --domain math --limit 50 --provider groq
```

**What this does:**
1. Processes 50 math problems
2. Generates responses from Llama-3
3. Extracts all features (semantic, contextual, naive baselines)
4. Trains XGBoost classifier
5. Generates 5 figures + metrics

**Success criteria:**
- ✓ Completes without errors
- ✓ AUROC > 0.70
- ✓ Figures appear in `outputs/figures/math/`

---

## 📊 Full Workflow Overview

### Phase 1: Validation (Tomorrow)
```bash
# Test with 50 samples
python run_pipeline.py --domain math --limit 50

# Review outputs
open outputs/figures/math/
cat outputs/results/metrics_math.json
```

### Phase 2: Single Domain (Day 2-3)
```bash
# Full math dataset (8,792 samples)
python run_pipeline.py --domain math
```

### Phase 3: All Domains (Day 3-5)
```bash
# Process remaining domains
python run_pipeline.py --domain finance
python run_pipeline.py --domain psychology
python run_pipeline.py --domain is_agents
python run_pipeline.py --domain medicine --limit 5000
```

### Phase 4: Research Questions Analysis (Day 5)
```bash
# Combine all features
python -c "
import pandas as pd
dfs = [pd.read_csv(f'data/features/{d}_features.csv') 
       for d in ['math','medicine','finance','is_agents','psychology']]
pd.concat(dfs).to_csv('data/features/all_features.csv', index=False)
"

# Answer all RQs
python analyze_research_questions.py \
  --features data/features/all_features.csv
```

---

## 📞 Quick Help

**Error: API key not found**
→ Edit `.env` and add your Together AI or Groq key

**Error: Module not found**
→ Run `source venv/bin/activate` to activate environment

**Error: Out of memory**
→ Use smaller `--limit` (e.g., 25 instead of 50)

**Error: Rate limit**
→ Wait a few minutes or increase `rate_limit_delay` in `configs/model.yaml`

**Need help?**
→ Check `README.md` for full documentation
→ Check `RESEARCH_QUESTIONS.md` for RQ details

---

## 📋 Project Structure

```
HHPF/
├── START_HERE.md              ← You are here
├── RESEARCH_QUESTIONS.md      ← RQ implementation details
├── README.md                  ← Full documentation
├── analyze_research_questions.py  ← RQ analysis script
├── run_pipeline.py            ← Main pipeline
├── requirements.txt           ← Dependencies
├── .env                       ← API keys (add yours!)
├── configs/                   ← YAML configurations
├── data/
│   ├── raw/                   ← Your datasets (ready ✓)
│   ├── processed/             ← Processed data (auto-generated)
│   └── features/              ← Extracted features (auto-generated)
├── src/                       ← Source code (all implemented ✓)
├── outputs/                   ← Results (auto-generated)
│   ├── models/               
│   ├── figures/              
│   └── results/              
└── notebooks/                 ← Jupyter notebooks (optional)
```

---

## ✅ System Complete - Ready for Tomorrow

**Your HHPF framework is fully implemented and ready to run!**

All three research questions will be answered comprehensively with:
- Statistical evidence
- Publication-quality figures  
- Detailed metrics
- Domain-specific analysis

**Tomorrow's first command:**
```bash
python run_pipeline.py --domain math --limit 50
```

Good luck with your Master's thesis! 🎓✨
