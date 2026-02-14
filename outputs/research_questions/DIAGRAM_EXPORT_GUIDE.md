# HHPF Diagram Export Guide

**Purpose:** Instructions for exporting Mermaid diagrams from `docs/PIPELINE_DIAGRAMS.md` to high-resolution PNG/PDF for thesis inclusion

**Date:** February 14, 2026  
**Status:** Ready for export

---

## Quick Reference

**Source:** [`docs/PIPELINE_DIAGRAMS.md`](../../docs/PIPELINE_DIAGRAMS.md)  
**Output:** `outputs/research_questions/figures/`  
**Format:** PNG (300 DPI) and PDF (vector)

---

## Method 1: Mermaid Live Editor (Recommended - Easiest)

### Steps:

1. **Open Mermaid Live Editor**
   - Visit: https://mermaid.live

2. **Copy Diagram Code**
   - Open `docs/PIPELINE_DIAGRAMS.md`
   - Copy the mermaid code block for your desired diagram (everything between \`\`\`mermaid and \`\`\`)

3. **Paste and Preview**
   - Paste into the editor
   - Diagram renders automatically

4. **Export**
   - Click "Actions" → "Download PNG" (for Word/Google Docs)
   - Click "Actions" → "Download SVG" (for vector graphics)
   - For PDF: Download SVG, then convert using `inkscape` or online converter

5. **Save to Correct Location**
   - Save to `outputs/research_questions/figures/`
   - Use naming convention: `{diagram_name}_{version}.png`

### Recommended Exports:

| Diagram | Filename | Priority |
|---------|----------|----------|
| Diagram 1: Complete Pipeline | `pipeline_complete.png` | **MUST** |
| Diagram 2: Research Methodology | `methodology_flow.png` | **MUST** |
| Diagram 3: Feature Engineering | `feature_architecture.png` | Recommended |
| Diagram 4: Ablation Design | `ablation_design.png` | Recommended |
| Diagram 5: Statistical Analysis | `statistical_analysis.png` | **MUST** |
| Diagram 6: Conceptual Overview | `conceptual_overview.png` | Optional |
| Diagram 7: End-to-End Flow | `end_to_end_flow.png` | Optional |
| Diagram 8: Data Flow | `data_flow_architecture.png` | Optional |

---

## Method 2: Mermaid CLI (For Batch Export)

### Prerequisites:

```bash
# Install mermaid-cli globally
sudo npm install -g @mermaid-js/mermaid-cli
```

### Extract Individual Diagrams:

First, create separate `.mmd` files for each diagram:

```bash
# Create temp directory
mkdir -p temp/diagrams

# Manually extract each mermaid block from PIPELINE_DIAGRAMS.md
# Save as temp/diagrams/diagram1.mmd, diagram2.mmd, etc.
```

### Export Commands:

```bash
# Diagram 1: Complete Pipeline
mmdc -i temp/diagrams/diagram1.mmd \
     -o outputs/research_questions/figures/pipeline_complete.png \
     -w 2400 -H 1600 -b transparent

# Diagram 2: Research Methodology
mmdc -i temp/diagrams/diagram2.mmd \
     -o outputs/research_questions/figures/methodology_flow.png \
     -w 2400 -H 2000 -b transparent

# Diagram 3: Feature Engineering
mmdc -i temp/diagrams/diagram3.mmd \
     -o outputs/research_questions/figures/feature_architecture.png \
     -w 2000 -H 1200 -b transparent

# Diagram 4: Ablation Design
mmdc -i temp/diagrams/diagram4.mmd \
     -o outputs/research_questions/figures/ablation_design.png \
     -w 2000 -H 1600 -b transparent

# Diagram 5: Statistical Analysis
mmdc -i temp/diagrams/diagram5.mmd \
     -o outputs/research_questions/figures/statistical_analysis.png \
     -w 2400 -H 1800 -b transparent

# Diagram 6: Conceptual Overview
mmdc -i temp/diagrams/diagram6.mmd \
     -o outputs/research_questions/figures/conceptual_overview.png \
     -w 1600 -H 1000 -b transparent

# Diagram 7: End-to-End Flow
mmdc -i temp/diagrams/diagram7.mmd \
     -o outputs/research_questions/figures/end_to_end_flow.png \
     -w 2400 -H 2400 -b transparent

# Diagram 8: Data Flow
mmdc -i temp/diagrams/diagram8.mmd \
     -o outputs/research_questions/figures/data_flow_architecture.png \
     -w 2400 -H 1400 -b transparent
```

### Export to PDF:

```bash
# Convert PNG to PDF (for LaTeX)
# Using ImageMagick:
convert outputs/research_questions/figures/pipeline_complete.png \
        -density 300 \
        outputs/research_questions/figures/pipeline_complete.pdf

# Or using Ghostscript for better quality:
gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -r300 \
   -sOutputFile=outputs/research_questions/figures/pipeline_complete.pdf \
   temp/input.png
```

---

## Method 3: Cursor Markdown Preview (Quick Screenshots)

### Steps:

1. **Open in Cursor**
   - Open `docs/PIPELINE_DIAGRAMS.md` in Cursor
   - Preview renders mermaid automatically

2. **Take Screenshot**
   - macOS: `Cmd+Shift+4` → drag to select diagram
   - Screenshot will be high quality

3. **Crop and Save**
   - Use Preview or any image editor to crop
   - Save to `outputs/research_questions/figures/`

**Pros:** Very fast, no dependencies  
**Cons:** Manual process, may have lower resolution than Method 1/2

---

## Method 4: Python Script (Automated)

### Using `mermaid.py` library:

```python
from mermaid import Mermaid

# Example for one diagram
diagram_code = """
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
"""

# Render to image
m = Mermaid(diagram_code)
m.to_png('outputs/research_questions/figures/example.png', width=2400, height=1600)
```

**Note:** Requires `mermaid` Python package and Playwright/Chromium installed.

---

## Resolution Guidelines

### For Digital Thesis (PDF):
- **Width:** 2400-2800px
- **Height:** 1600-2400px (depending on diagram complexity)
- **Format:** PNG (300 DPI) or PDF (vector)
- **Background:** Transparent or white

### For Printed Thesis:
- **Minimum:** 300 DPI
- **Recommended:** 600 DPI for complex diagrams
- **Format:** PDF (vector preferred)

### File Size:
- PNG files will be 1-5 MB each
- PDF files will be smaller (<1 MB)
- Total for 8 diagrams: ~10-20 MB

---

## Thesis Integration Checklist

### Required for Results Chapter:
- [ ] `pipeline_complete.png` (Diagram 1)
- [ ] `methodology_flow.png` (Diagram 2)
- [ ] `statistical_analysis.png` (Diagram 5)

### Recommended for Methods/Results:
- [ ] `feature_architecture.png` (Diagram 3)
- [ ] `ablation_design.png` (Diagram 4)

### Optional for Appendix:
- [ ] `conceptual_overview.png` (Diagram 6)
- [ ] `end_to_end_flow.png` (Diagram 7)
- [ ] `data_flow_architecture.png` (Diagram 8)

---

## LaTeX Integration

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/pipeline_complete.pdf}
  \caption{Complete HHPF technical pipeline showing data flow from raw datasets through feature extraction, model training, and evaluation across five domains. Each domain is processed independently to prevent cross-domain contamination.}
  \label{fig:pipeline_complete}
\end{figure}
```

---

## Word/Google Docs Integration

1. Insert → Image → Upload from computer
2. Select exported PNG file
3. Resize to fit page width (typically 6-7 inches)
4. Add caption below:
   - **Figure X:** Complete HHPF technical pipeline...
5. Center align

---

## Quality Checklist

Before using in thesis:

- [ ] Text is readable at print size
- [ ] No pixelation or blurriness
- [ ] Colors/contrast work in grayscale (for printing)
- [ ] Labels and arrows are clear
- [ ] Consistent styling across all diagrams
- [ ] File size is reasonable (<5 MB per image)

---

## Troubleshooting

### Issue: "Diagram too large to render"
**Solution:** Use Method 2 (CLI) with custom dimensions

### Issue: "Text is too small"
**Solution:** Increase export width/height, or simplify diagram

### Issue: "Colors don't print well"
**Solution:** Use grayscale or high-contrast color scheme

### Issue: "Arrows overlap with text"
**Solution:** Adjust mermaid code with more spacing or reorder nodes

---

## Recommended Workflow

**For Thesis Submission:**

1. **Use Method 1 (Mermaid Live)** for quick, high-quality exports
2. Export **3 must-have diagrams** first (1, 2, 5)
3. Export remaining diagrams as needed
4. Save both PNG and PDF versions
5. Test in your thesis template to ensure proper rendering
6. Keep source code in `docs/PIPELINE_DIAGRAMS.md` for future edits

**Time Estimate:**
- 5-10 minutes per diagram using Method 1
- ~1 hour total for all 8 diagrams

---

## Current Status

- ✅ Diagrams created in `docs/PIPELINE_DIAGRAMS.md`
- ⏳ Awaiting manual export (use Method 1 recommended)
- ⏳ High-resolution PNG/PDF generation

**Next Steps:**
1. Visit https://mermaid.live
2. Export 3 priority diagrams (1, 2, 5)
3. Save to `outputs/research_questions/figures/`
4. Verify quality in thesis template

---

**Guide Created:** February 14, 2026  
**Last Updated:** February 14, 2026  
**Status:** Ready for use
