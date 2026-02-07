# Research Action Plan – Complete Remaining Domains

**Date**: February 2026  
**Goal**: Finish medicine (reprocess with fix) and HalluMix (IS/Agents) so all 5 domains have valid results.

---

## Current Status

| Domain      | Samples | AUROC   | Status        | Action |
|------------|--------|---------|---------------|--------|
| Math       | 542    | 0.79    | Valid         | None   |
| Finance    | 150    | 0.68    | Valid         | None   |
| Psychology | 500    | 0.71    | Valid (fixed)  | None   |
| Medicine   | 500    | **0.60** | ✅ Valid (reprocessed) | None — see `MEDICINE_DOMAIN_RESULTS.md` |
| IS/Agents (HalluMix) | 0 | -       | Not run       | Download + run with Groq (continue here) |

---

## Order of Execution

```mermaid
flowchart LR
    A[1. Fix is_agents pipeline] --> B[2. Medicine reprocess]
    B --> C[3. Download HalluMix]
    C --> D[4. HalluMix 100-sample test]
    D --> E[5. HalluMix full or 500]
```

1. **Fix is_agents (HalluMix) in pipeline** – Preserve and use `existing_label` in process + labeling (one-time code fix).
2. **Medicine reprocess** – Run `reprocess_cached_responses.py --domain medicine` (reuse 500 cached responses, ~30–45 min).
3. **Download HalluMix** – `python scripts/prepare_datasets.py --download-hallumix` (needs network).
4. **HalluMix 100-sample test** – `run_pipeline.py --domain is_agents --limit 100 --provider groq` to validate before scaling.
5. **HalluMix scale** – If 100-sample looks good, run 500 or full dataset with Groq.

---

## Step 1: Fix is_agents (HalluMix) in Pipeline

**Issue**: HalluMix has a pre-existing `hallucination_label`. The pipeline must:
- Keep `existing_label` in the processed CSV.
- Pass `existing_label` into the labeler in `label_responses.py` so `ExistingLabelLabeler` uses it instead of inferring labels.

**Changes**:
1. **`src/data_preparation/process_datasets.py`**  
   - Add `existing_label` to `output_columns` so it is written to `*_processed.csv` when present (e.g. for is_agents).
2. **`src/data_preparation/label_responses.py`**  
   - When calling the labeler, pass `existing_label=row.get('existing_label')` (or equivalent) for is_agents so `ExistingLabelLabeler` uses it.

**Check**: After running step 3–4, processed CSV for is_agents should have `existing_label`, and labeling should report “existing_label” as method.

---

## Step 2: Medicine Reprocess (No New API)

**Command**:
```bash
cd /Users/aviv.gross/HHPF
source venv/bin/activate  # or: venv/bin/python
python reprocess_cached_responses.py --domain medicine
```

**What it does**:
- Uses existing `medicine_*_responses.pkl` (500 files).
- Re-labels with current logic.
- Stratified split on `hallucination_label`.
- Extracts features, trains model, saves metrics.

**Time**: ~30–45 min (feature extraction is the slow part).  
**Cost**: $0 (no API calls).

**Success criteria**:
- Train/test hallucination rate gap &lt; 5%.
- Test AUROC &gt; 0.55 (improvement over ~0.51).
- No crashes; metrics and model saved.

**If it crashes (e.g. on viz)**: Model and metrics are usually already saved; you can regenerate plots with:
```bash
python generate_clean_viz.py --domain medicine
```

---

## Step 3: Download HalluMix

**Command**:
```bash
cd /Users/aviv.gross/HHPF
python scripts/prepare_datasets.py --download-hallumix
```

**Requires**: Network; `datasets` (HuggingFace) installed.  
**Output**: `data/raw/hallumix.csv` (question, answer, hallucination_label, etc.).

**Check**:
```bash
head -2 data/raw/hallumix.csv
wc -l data/raw/hallumix.csv
```

---

## Step 4: HalluMix 100-Sample Test (Groq)

**Command**:
```bash
cd /Users/aviv.gross/HHPF
python run_pipeline.py --domain is_agents --limit 100 --provider groq
```

**Requires**: `GROQ_API_KEY` in `.env`.  
**Steps**: Data prep (with `existing_label`) → inference (Groq) → label (use existing_label) → stratified split → features → train → evaluate.

**Time**: ~15–30 min (100 samples).  
**Success criteria**:
- Processed CSV has `existing_label`.
- Stratification gap &lt; 5%.
- Test AUROC &gt; 0.55.
- Both classes present in train and test.

**If “existing_label” is missing or wrong**: Re-check step 1 (process_datasets + label_responses).

---

## Step 5: HalluMix at Scale (500 or Full)

**After 100-sample looks good**:

**Option A – 500 samples** (recommended for parity with other domains):
```bash
python run_pipeline.py --domain is_agents --limit 500 --provider groq
```

**Option B – Full dataset** (~2.5k+ samples, longer run):
```bash
python run_pipeline.py --domain is_agents --provider groq
```

**Note**: Groq rate limits may apply; pipeline has retries. Run in a stable session (e.g. `nohup` or `tmux` if on a server).

---

## Checklist Summary

- [ ] **Step 1**: Code fix – `existing_label` in process_datasets + label_responses for is_agents.
- [x] **Step 2**: Medicine – `python reprocess_cached_responses.py --domain medicine` ✅ (AUROC 0.60, 0% gap).
- [ ] **Step 3**: Download – `python scripts/prepare_datasets.py --download-hallumix`.
- [ ] **Step 4**: HalluMix 100 – `python run_pipeline.py --domain is_agents --limit 100 --provider groq`.
- [ ] **Step 5**: HalluMix 500/full – run with `--limit 500` or no limit.

---

## After All Domains Are Done

- Regenerate any missing figures: `python generate_clean_viz.py --domain medicine` and `--domain is_agents` if needed.
- Update `EXPERIMENT_LOG.md` and project docs with final AUROCs and sample counts.
- Cross-domain comparison: AUROC and calibration by domain (math, finance, psychology, medicine, is_agents).
- Tie results back to `RESEARCH_QUESTIONS.md` (e.g. RQ1: semantic entropy; RQ2: domain variability).

---

## Quick Reference Commands

```bash
# 1. Medicine reprocess (no API)
python reprocess_cached_responses.py --domain medicine

# 2. Download HalluMix
python scripts/prepare_datasets.py --download-hallumix

# 3. HalluMix 100 test
python run_pipeline.py --domain is_agents --limit 100 --provider groq

# 4. HalluMix 500
python run_pipeline.py --domain is_agents --limit 500 --provider groq

# 5. Regenerate viz if needed
python generate_clean_viz.py --domain medicine
python generate_clean_viz.py --domain is_agents
```
