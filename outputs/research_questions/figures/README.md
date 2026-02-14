# HHPF Thesis Figures

**Purpose:** High-resolution figures for thesis Results chapter  
**Date:** February 14, 2026

---

## Directory Contents

This directory contains publication-quality figures for the Master's thesis:

### Research Question Figures (Generated)

✅ **Already Generated:**

1. `rq1_ablation_comparison.png` - RQ1: Hybrid vs Baseline feature comparison
2. `rq1_ablation_comparison.pdf` - PDF version
3. `rq2_semantic_vs_naive.png` - RQ2: Semantic vs Naive confidence
4. `rq2_semantic_vs_naive.pdf` - PDF version
5. `rq3a_hallucination_rates.png` - RQ3a: Per-domain hallucination rates
6. `rq3a_hallucination_rates.pdf` - PDF version
7. `rq3b_domain_auroc_variance.png` - RQ3b: Cross-domain AUROC variance
8. `rq3b_domain_auroc_variance.pdf` - PDF version
9. `rq3c_feature_importance_heatmap.png` - RQ3c: Feature importance variability
10. `rq3c_feature_importance_heatmap.pdf` - PDF version

### Pipeline Diagrams (To Be Exported)

⏳ **Pending Export from `docs/PIPELINE_DIAGRAMS.md`:**

**Priority 1 (Must Have):**
- `pipeline_complete.png` - Complete technical pipeline (Diagram 1)
- `methodology_flow.png` - Research methodology flow (Diagram 2)
- `statistical_analysis.png` - Statistical analysis workflow (Diagram 5)

**Priority 2 (Recommended):**
- `feature_architecture.png` - Feature engineering architecture (Diagram 3)
- `ablation_design.png` - Per-domain ablation design (Diagram 4)

**Priority 3 (Optional):**
- `conceptual_overview.png` - High-level overview (Diagram 6)
- `end_to_end_flow.png` - Complete research flow (Diagram 7)
- `data_flow_architecture.png` - Data flow architecture (Diagram 8)

---

## Export Instructions

**To export pipeline diagrams:**

1. See comprehensive guide: [`DIAGRAM_EXPORT_GUIDE.md`](../DIAGRAM_EXPORT_GUIDE.md)
2. **Recommended method:** Use https://mermaid.live
   - Copy diagram code from `docs/PIPELINE_DIAGRAMS.md`
   - Paste into Mermaid Live Editor
   - Export as PNG (high resolution) or SVG
   - Save to this directory

3. **Alternative:** Use Cursor markdown preview + screenshot

**Time estimate:** 5-10 minutes per diagram

---

## Thesis Integration

### Results Chapter (Section 4)

**Section 4.1 - Experimental Setup:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/pipeline_complete.pdf}
  \caption{Complete HHPF technical pipeline...}
  \label{fig:pipeline}
\end{figure}
```

**Section 4.2 - Per-Domain Results:**
- Use `rq3a_hallucination_rates.pdf`
- Use `rq3b_domain_auroc_variance.pdf`

**Section 4.3 - Feature Analysis:**
- Use `rq1_ablation_comparison.pdf`
- Use `rq2_semantic_vs_naive.pdf`
- Use `rq3c_feature_importance_heatmap.pdf`

**Section 4.4 - Methodology:**
- Use `methodology_flow.pdf`
- Use `ablation_design.pdf`
- Use `statistical_analysis.pdf`

---

## Figure Specifications

**Resolution:**
- PNG: 300 DPI minimum
- Size: 2000-2800px width for full-page figures
- Size: 1200-1600px width for half-page figures

**Format:**
- PNG for Word/Google Docs
- PDF for LaTeX (vector graphics, best quality)
- SVG for web/presentations (optional)

**Naming Convention:**
- Lowercase with underscores
- Descriptive names (e.g., `rq1_ablation_comparison.png`)
- Include version suffix if multiple iterations (e.g., `_v2.png`)

---

## Quality Checklist

Before using in thesis:

- [ ] All text is readable when printed at thesis size
- [ ] No pixelation or blurriness
- [ ] Colors work in grayscale (for printing)
- [ ] Consistent font sizes across all figures
- [ ] Axes labels are clear and complete
- [ ] Legend is positioned correctly
- [ ] File size is reasonable (<5 MB)

---

## Current Status

**Generated (10 figures):**
- ✅ All RQ1, RQ2, RQ3 statistical figures
- ✅ Both PNG and PDF versions
- ✅ Thesis-ready quality

**Pending Export (3-8 diagrams):**
- ⏳ Pipeline and methodology diagrams
- ⏳ Waiting for manual export from Mermaid Live
- ⏳ See `DIAGRAM_EXPORT_GUIDE.md` for instructions

**Total Figures for Thesis:** 13-18 (10 generated + 3-8 pending)

---

## File Organization

```
outputs/research_questions/figures/
├── README.md                              (this file)
├── rq1_ablation_comparison.png            ✅ generated
├── rq1_ablation_comparison.pdf            ✅ generated
├── rq2_semantic_vs_naive.png              ✅ generated
├── rq2_semantic_vs_naive.pdf              ✅ generated
├── rq3a_hallucination_rates.png           ✅ generated
├── rq3a_hallucination_rates.pdf           ✅ generated
├── rq3b_domain_auroc_variance.png         ✅ generated
├── rq3b_domain_auroc_variance.pdf         ✅ generated
├── rq3c_feature_importance_heatmap.png    ✅ generated
├── rq3c_feature_importance_heatmap.pdf    ✅ generated
├── pipeline_complete.png                  ⏳ to be exported
├── methodology_flow.png                   ⏳ to be exported
├── statistical_analysis.png               ⏳ to be exported
├── feature_architecture.png               ⏳ optional
├── ablation_design.png                    ⏳ optional
└── ...                                    (other optional diagrams)
```

---

**Last Updated:** February 14, 2026  
**Status:** 10/13 minimum figures ready for thesis
