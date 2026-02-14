# HHPF Diagrams Index

**Purpose:** Quick reference for all diagrams in the HHPF project  
**Date:** February 14, 2026

---

## Available Diagrams

### 📊 Pipeline Diagrams (A-Z Workflow)

**Source:** [`PIPELINE_DIAGRAMS.md`](PIPELINE_DIAGRAMS.md)

| # | Diagram Name | Purpose | Thesis Section |
|---|--------------|---------|----------------|
| 1 | Complete HHPF Pipeline | Show technical workflow (data → results) | Results 4.1 |
| 2 | Research Methodology Flow | Phase A → Phase B analysis | Results 4.2 |
| 3 | Feature Engineering Architecture | 3 feature categories explained | Methods |
| 4 | Ablation Study Design | Per-domain approach | Methods |
| 5 | Cross-Domain Statistical Analysis | How RQ are answered | Results 4.2 |
| 6 | Conceptual Overview | High-level summary | Introduction |
| 7 | End-to-End Research Flow | Complete research loop | Appendix |
| 8 | Data Flow Architecture | System storage organization | Appendix |

**Format:** Mermaid (can export to PNG/PDF)  
**Status:** ✅ Created, ⏳ awaiting export

**Export Instructions:** [`../outputs/research_questions/DIAGRAM_EXPORT_GUIDE.md`](../outputs/research_questions/DIAGRAM_EXPORT_GUIDE.md)

---

### 📈 Statistical Figures (Generated)

**Source:** `outputs/research_questions/figures/`

| # | Figure Name | Research Question | Format |
|---|-------------|-------------------|--------|
| 1 | RQ1 Ablation Comparison | RQ1: Hybrid vs Baseline | PNG + PDF |
| 2 | RQ2 Semantic vs Naive | RQ2: Semantic uncertainty | PNG + PDF |
| 3 | RQ3a Hallucination Rates | RQ3: Domain differences | PNG + PDF |
| 4 | RQ3b Domain AUROC Variance | RQ3: Performance variance | PNG + PDF |
| 5 | RQ3c Feature Importance Heatmap | RQ3: Feature variability | PNG + PDF |

**Status:** ✅ Generated and thesis-ready

---

### 📉 Per-Domain Visualizations

**Source:** `outputs/figures/{domain}/`

For each of 5 domains (math, is_agents, psychology, medicine, finance):

| Figure Type | Description |
|-------------|-------------|
| ROC Curve | AUROC performance |
| Calibration Plot | ECE and reliability |
| Confusion Matrix | Classification results |
| Feature Importance | Top 20 features |
| ARC Plot | Accuracy-Rejection Curve |

**Status:** ✅ Generated for all 5 domains

---

## Quick Access

### For Thesis Results Chapter:

**Priority 1 (Must Include):**
1. Pipeline Diagram 1 (Complete Pipeline) - Shows technical workflow
2. Pipeline Diagram 2 (Methodology) - Shows Phase A → B approach
3. RQ1 Ablation Figure - Feature comparison
4. RQ3a Hallucination Rates - Domain differences
5. RQ3b AUROC Variance - Performance across domains

**Priority 2 (Recommended):**
6. Pipeline Diagram 5 (Statistical Analysis) - How RQ answered
7. RQ2 Semantic vs Naive - Confidence comparison
8. RQ3c Feature Importance Heatmap - Variability analysis
9. Select per-domain figures (e.g., Math ROC, IS Agents calibration)

**Priority 3 (Optional):**
10. Pipeline Diagrams 3, 4, 6, 7, 8 - Additional context
11. Remaining per-domain figures

---

## Export Status

### ✅ Ready for Use:
- All 10 statistical figures (PNG + PDF)
- All 25 per-domain figures (PNG + PDF)

### ⏳ Pending Export:
- 8 pipeline diagrams (need manual export from Mermaid Live)
- Estimated time: 15-20 minutes for 3 priority diagrams
- Estimated time: 40-60 minutes for all 8 diagrams

**See:** [`QUICK_EXPORT.md`](../outputs/research_questions/figures/QUICK_EXPORT.md) for fastest export method

---

## File Locations

```
HHPF/
├── docs/
│   ├── PIPELINE_DIAGRAMS.md           ← 8 mermaid diagrams (source)
│   └── DIAGRAMS_INDEX.md              ← This file
├── outputs/
│   ├── research_questions/
│   │   ├── DIAGRAM_EXPORT_GUIDE.md    ← Export instructions
│   │   └── figures/
│   │       ├── QUICK_EXPORT.md        ← Fast export guide
│   │       ├── rq1_*.png/pdf          ← Statistical figures ✅
│   │       ├── rq2_*.png/pdf          ← Statistical figures ✅
│   │       └── rq3*.png/pdf           ← Statistical figures ✅
│   └── figures/
│       ├── math/                      ← Per-domain figures ✅
│       ├── is_agents/                 ← Per-domain figures ✅
│       ├── psychology/                ← Per-domain figures ✅
│       ├── medicine/                  ← Per-domain figures ✅
│       └── finance/                   ← Per-domain figures ✅
```

---

## Diagram Count Summary

| Category | Count | Status |
|----------|-------|--------|
| Pipeline Diagrams | 8 | ✅ Created, ⏳ Need export |
| Statistical Figures | 10 | ✅ Generated |
| Per-Domain Figures | 25 | ✅ Generated |
| **Total** | **43** | **35 ready, 8 pending** |

---

## Next Steps

1. **Export Priority 1 Diagrams** (3 diagrams, ~15 min)
   - Use Mermaid Live Editor: https://mermaid.live
   - Follow [`QUICK_EXPORT.md`](../outputs/research_questions/figures/QUICK_EXPORT.md)

2. **Verify Quality**
   - Check readability at print size
   - Ensure grayscale compatibility

3. **Integrate into Thesis**
   - Place diagrams in appropriate sections
   - Add figure captions and labels

---

**Created:** February 14, 2026  
**Last Updated:** February 14, 2026  
**Status:** Index complete, awaiting diagram export
