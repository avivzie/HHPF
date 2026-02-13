# Repository Cleanup Checklist
**Priority:** LOW - Execute AFTER research analysis is complete  
**Goal:** Make repo publication-ready and professional

---

## 🎯 Execute This AFTER:
- ✅ All 5 domains validated
- ✅ RQ1, RQ2, RQ3 analysis complete
- ✅ Results chapter drafted
- ✅ Thesis finalized

---

## High Priority Cleanup

### 1. Remove Temporary/Debug Files
- [ ] Remove `.matplotlib_cache/` directory
- [ ] Clean up any `.DS_Store` files
- [ ] Remove test scripts in root directory (if any)
- [ ] Clean up `cache/` directory (keep structure, remove old cached responses if needed)

### 2. Organize Documentation
- [ ] Move all research logs to `docs/research_logs/` (already done)
- [ ] Create main `README.md` with project overview
- [ ] Add `INSTALLATION.md` with setup instructions
- [ ] Create `USAGE.md` with pipeline execution guide
- [ ] Keep `CHANGELOG.md` but archive old entries

### 3. Results Directory Cleanup
- [ ] Keep validation reports (DOMAIN_VALIDATION_REPORT.md, etc.)
- [ ] Archive intermediate verification JSON files to subdirectory
- [ ] Ensure all final metrics are in `outputs/results/metrics_*.json`
- [ ] Keep only final figures (PNG + PDF for each domain)

### 4. Code Cleanup
- [ ] Remove unused imports across all Python files
- [ ] Add docstrings to any undocumented functions
- [ ] Remove commented-out code blocks
- [ ] Ensure consistent code formatting (run black/autopep8)
- [ ] Remove any hardcoded paths (use config files)

### 5. Configuration Files
- [ ] Review all YAML configs for obsolete parameters
- [ ] Document all config options in comments
- [ ] Create `configs/README.md` explaining each config file

---

## Medium Priority Cleanup

### 6. Data Directory Organization
- [ ] Create `data/README.md` explaining structure
- [ ] Document expected format for raw data files
- [ ] Remove any duplicate or superseded data files
- [ ] Consider compressing large raw data files

### 7. Scripts Organization
- [ ] Move utility scripts to `scripts/` directory
- [ ] Remove one-off debugging scripts
- [ ] Add docstrings/headers to all scripts
- [ ] Create `scripts/README.md` with usage examples

### 8. Outputs Organization
```
outputs/
├── figures/          # Keep all domain figures
├── models/           # Keep trained models (or .gitignore them if too large)
├── results/          # Keep final metrics + validation reports
└── archive/          # Move intermediate results here
```

### 9. Tests (if time permits)
- [ ] Add basic unit tests for key functions
- [ ] Create `tests/` directory
- [ ] Add `pytest` configuration
- [ ] Document test execution in README

---

## Low Priority (Polish)

### 10. Main README.md Structure
```markdown
# HHPF: Hallucination Detection via Hybrid Probabilistic Features

## Overview
Brief description of the project

## Key Results
Summary table of 5 domains with AUROC

## Installation
Quick start guide

## Usage
How to run the pipeline

## Repository Structure
Directory tree with explanations

## Citation
How to cite the work

## License
MIT or appropriate license
```

### 11. GitHub/Publication Prep
- [ ] Add LICENSE file
- [ ] Create `.gitignore` for Python, data files, model files
- [ ] Add badges to README (Python version, license, etc.)
- [ ] Create `CONTRIBUTING.md` if making repo public
- [ ] Add requirements.txt with pinned versions

### 12. Documentation Polish
- [ ] Spell check all markdown files
- [ ] Fix broken links in documentation
- [ ] Ensure consistent formatting across docs
- [ ] Add table of contents to long documents

### 13. Notebook Examples (Optional)
- [ ] Create Jupyter notebook showing pipeline walkthrough
- [ ] Add visualization notebook for results
- [ ] Document notebook requirements

---

## Files to Definitely Keep

### Core Code
- `src/` - All source code
- `configs/` - All configuration files
- `run_pipeline.py` - Main execution script

### Documentation
- `docs/RESEARCH_METHODOLOGY.md`
- `docs/research_logs/EXPERIMENT_LOG.md`
- All domain validation documents
- `CHANGELOG.md`

### Results
- `outputs/results/metrics_*.json` (5 domains)
- `outputs/results/DOMAIN_VALIDATION_REPORT.md`
- `outputs/results/VERIFICATION_SUMMARY.md`
- `outputs/figures/**/*.png` and `*.pdf` (50 figures)

### Models
- `outputs/models/xgboost_*.pkl` (5 models)
- Consider .gitignore if repo goes public (large files)

---

## Files to Consider Removing/Archiving

### Temporary Files
- `.matplotlib_cache/`
- `__pycache__/` directories (add to .gitignore)
- `.DS_Store` files
- Intermediate test outputs

### Superseded Documentation
- Old research logs before final validation
- Draft documents no longer relevant
- Duplicate checklists (consolidate)

### Large Data Files
- Raw datasets (document how to obtain instead)
- Response cache files (too large for repo)
- Consider hosting on external storage if needed

---

## Git Cleanup Commands (Execute Last)

```bash
# Remove cached files from git
git rm -r --cached .matplotlib_cache/
git rm --cached **/.DS_Store

# Add proper .gitignore
cat >> .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data
cache/
*.pkl
data/raw/**/*.csv

# OS
.DS_Store
.matplotlib_cache/

# IDE
.vscode/
.idea/
*.swp

# Outputs (keep structure, ignore large files)
outputs/models/*.pkl
EOF

# Commit cleanup
git add .
git commit -m "Repository cleanup for publication"
```

---

## Final Quality Check

- [ ] Run pipeline end-to-end on one domain to verify nothing broke
- [ ] Verify all imports work in clean environment
- [ ] Check that README instructions are accurate
- [ ] Ensure no sensitive information (API keys, paths)
- [ ] Test installation instructions in fresh environment
- [ ] Get colleague/peer review if possible

---

## Publication Readiness Checklist

- [ ] README is clear and professional
- [ ] All code is documented
- [ ] Results are reproducible from provided scripts
- [ ] License added
- [ ] Citation information provided
- [ ] No TODO comments in production code
- [ ] All hardcoded values moved to configs
- [ ] Repo looks professional on GitHub

---

**Status:** 🔴 Not Started - Execute after research complete  
**Priority:** Clean enough for thesis submission, polish for paper submission  
**Estimated Time:** 4-8 hours of cleanup work

**Note:** Don't let perfect be the enemy of good. Main goal is clear, reproducible research. Polish can always come later if needed for publication.
