# Project Cleanup Plan

**Date:** February 14, 2026  
**Purpose:** Archive temporary, intermediate, and unused artifacts

## Files to Archive

### 1. data/processed/ (2 files)

#### Archive:
- ❌ `medicine_processed_337.csv` (331 lines, 96K)
  - **Reason:** Testing checkpoint with only 337 samples
  - **Final version:** `medicine_processed.csv` (4,688 lines, 427K)

- ❌ `psychology_processed_stratified.csv` (17,930 lines, 2.2M)
  - **Reason:** OLD full dataset before stratification
  - **Final version:** `psychology_processed.csv` (501 lines, 102K - stratified sample)

#### Keep:
- ✅ `finance_processed.csv` (23,349 lines)
- ✅ `is_agents_processed.csv` (9,396 lines)
- ✅ `math_processed.csv` (8,793 lines)
- ✅ `medicine_processed.csv` (4,688 lines)
- ✅ `psychology_processed.csv` (501 lines)

---

### 2. data/features/ (5 files)

#### Archive:
- ❌ `psychology_features_complete.csv` (2.3M, Feb 7)
  - **Reason:** Duplicate/backup version
  
- ❌ `psychology_features_fixed.csv` (2.3M, Feb 7)
  - **Reason:** Intermediate fix version
  
- ❌ `psychology_features_new.csv` (2.3M, Feb 7)
  - **Reason:** Intermediate version during development
  
- ❌ `psychology_features_OLD_BACKUP.csv` (2.3M, Feb 7)
  - **Reason:** Explicitly marked as OLD_BACKUP

- ❌ `responses_finance_processed.csv` (1.2M)
- ❌ `responses_is_agents_processed.csv` (1.8M)
- ❌ `responses_math_processed.csv` (814K)
- ❌ `responses_medicine_processed.csv` (2.0M)
- ❌ `responses_psychology_processed.csv` (2.2M)
  - **Reason:** Intermediate files (responses before feature extraction)
  - **Note:** Full features already in `{domain}_features.csv`

#### Keep:
- ✅ `finance_features.csv` (1.3M)
- ✅ `is_agents_features.csv` (1.9M)
- ✅ `math_features.csv` (796K)
- ✅ `medicine_features.csv` (2.1M)
- ✅ `psychology_features.csv` (2.3M)

---

### 3. data/raw/ (1 file)

#### Archive:
- ❌ `financebench_sample_150.csv`
  - **Reason:** OLD finance dataset (only 150 samples)
  - **Replaced by:** TAT-QA dataset (23,349 samples)

#### Keep:
- ✅ `gsm8k.csv` - Math dataset
- ✅ `hallumix.csv` - IS Agents dataset
- ✅ `TruthfulQA.csv` - Psychology dataset
- ✅ `med_halt.csv` - Medicine dataset
- ✅ `tatqa.csv` - Finance dataset (current)
- ✅ `TATQA/` - TAT-QA JSON source files
- ✅ `GSM8K/` - GSM8K CSV source files

---

### 4. System Files (cleanup, not archive)

#### Delete:
- 🗑️ `.DS_Store` files (Mac system files)
  - Found in: root, cache/, cache/responses/, docs/, outputs/
  - **Reason:** System metadata, should be in .gitignore

---

### 5. Already Archived (no action needed)

These are already in appropriate archive folders:
- ✅ `outputs/results/archive/` - Old validation reports (16 files)
- ✅ `outputs/models/archive_old_experiments/` - Old models (5 files)
- ✅ `scripts/archive_old_utilities/` - Old scripts (4 files)

---

## Files to Keep (Essential for Thesis)

### data/processed/ (5 files)
- Final processed datasets for all 5 domains

### data/features/ (5 files)
- Final feature CSV files for all 5 domains

### data/raw/ (6 files + 2 dirs)
- Original source datasets

### outputs/models/ (5 files)
- Final trained models

### outputs/results/ (7 files)
- Final metrics and summaries

### outputs/ablation/ (10 files)
- Ablation study results

### outputs/research_questions/ (5 CSVs + 1 JSON + figures/)
- Research question results

### outputs/figures/ (5 domain folders)
- Per-domain visualizations

---

## Archive Structure

```
data/
├── processed/
│   └── archive_intermediate/
│       ├── medicine_processed_337.csv
│       └── psychology_processed_stratified.csv
│
├── features/
│   └── archive_intermediate/
│       ├── psychology_features_complete.csv
│       ├── psychology_features_fixed.csv
│       ├── psychology_features_new.csv
│       ├── psychology_features_OLD_BACKUP.csv
│       ├── responses_finance_processed.csv
│       ├── responses_is_agents_processed.csv
│       ├── responses_math_processed.csv
│       ├── responses_medicine_processed.csv
│       └── responses_psychology_processed.csv
│
└── raw/
    └── archive_old_datasets/
        └── financebench_sample_150.csv
```

---

## Space Savings

### Before Cleanup:
- data/processed/: ~14.0 MB
- data/features/: ~16.9 MB
- Total: ~30.9 MB

### After Cleanup:
- data/processed/: ~11.7 MB (-2.3 MB)
- data/features/: ~10.2 MB (-6.7 MB)
- Total: ~21.9 MB

**Space Saved:** ~9 MB (29% reduction)

---

## Commands to Execute

```bash
# Create archive directories
mkdir -p data/processed/archive_intermediate
mkdir -p data/features/archive_intermediate
mkdir -p data/raw/archive_old_datasets

# Archive data/processed
mv data/processed/medicine_processed_337.csv data/processed/archive_intermediate/
mv data/processed/psychology_processed_stratified.csv data/processed/archive_intermediate/

# Archive data/features
mv data/features/psychology_features_*.csv data/features/archive_intermediate/
mv data/features/responses_*_processed.csv data/features/archive_intermediate/

# Archive data/raw
mv data/raw/financebench_sample_150.csv data/raw/archive_old_datasets/

# Clean .DS_Store files
find . -name ".DS_Store" -not -path "./venv/*" -delete

# Create README files in archives
# (separate step)
```

---

## Verification After Cleanup

```bash
# Verify only 5 processed files remain
ls -1 data/processed/*.csv | wc -l  # Should be 5

# Verify only 5 feature files remain
ls -1 data/features/*_features.csv | wc -l  # Should be 5

# Verify archives created
ls data/processed/archive_intermediate/
ls data/features/archive_intermediate/
ls data/raw/archive_old_datasets/
```

---

## Impact Assessment

### ✅ Safe to Archive (No Dependencies)
- All archived files are intermediate/testing versions
- Final versions exist and are actively used
- No scripts reference archived files

### ⚠️ Not Archived (Need to Keep)
- `cache/responses/` - API response cache (saves $5-10 in API costs)
- `outputs/results/archive/` - Already archived, documented
- `venv/` - Python virtual environment

---

## Next Steps

1. Execute cleanup commands
2. Create README files in archive directories
3. Verify project still works (run quick test)
4. Update .gitignore to exclude .DS_Store
5. Commit cleanup changes

---

**Status:** Ready to execute
