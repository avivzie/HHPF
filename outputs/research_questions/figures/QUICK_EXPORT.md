# Quick Export Instructions

**Goal:** Export 3 priority diagrams for thesis in ~15 minutes

---

## Step-by-Step Guide

### 1. Open Mermaid Live Editor

Go to: **https://mermaid.live**

### 2. Export Diagram 1: Complete Pipeline

**Source:** Open `docs/PIPELINE_DIAGRAMS.md` and find **Diagram 1: Complete HHPF Technical Pipeline**

**Copy this code block:**

```mermaid
flowchart TD
    subgraph input [Input Data]
        D1[Math: GSM8K<br/>n=542]
        D2[IS Agents: HalluMix<br/>n=500]
        D3[Psychology: TruthfulQA<br/>n=500]
        D4[Medicine: Med-HALT<br/>n=500]
        D5[Finance: TAT-QA<br/>n=500]
    end
    
    subgraph processing [Step 1: Data Processing]
        DP[Process Dataset<br/>Clean & Format]
    end
    
    subgraph inference [Step 2: Response Generation]
        LLM[LLM Inference<br/>Llama-3.1-8B<br/>5 stochastic samples]
        CACHE1[(Cached Responses<br/>~2,500 prompts)]
    end
    
    subgraph labeling [Step 3: Ground Truth Labeling]
        GT[Compute Ground Truth<br/>Domain-Specific Methods]
        SPLIT[Stratified Split<br/>80% Train / 20% Test]
    end
    
    subgraph features [Step 4: Feature Extraction]
        FSEM[Semantic Features<br/>Entropy, Energy, Clusters]
        FCTX[Contextual Features<br/>Entities, Complexity, Rarity]
        FNAV[Naive Features<br/>MaxProb, Perplexity, LogProbs]
        FEAT[(Feature Matrix<br/>41-48 features)]
    end
    
    subgraph training [Step 5: Model Training]
        XGB[XGBoost Classifier<br/>Optuna Tuning<br/>20 trials]
        MODEL[(Trained Model<br/>Per Domain)]
    end
    
    subgraph evaluation [Step 6: Evaluation]
        METRICS[Calculate Metrics<br/>AUROC, Accuracy, ECE]
        VIZ[Generate Figures<br/>ROC, ARC, Calibration]
    end
    
    subgraph outputs [Per-Domain Results]
        R1[Math: AUROC 0.797]
        R2[IS Agents: AUROC 0.703]
        R3[Psychology: AUROC 0.671]
        R4[Finance: AUROC 0.632]
        R5[Medicine: AUROC 0.619]
    end
    
    D1 --> DP
    D2 --> DP
    D3 --> DP
    D4 --> DP
    D5 --> DP
    
    DP --> LLM
    LLM --> CACHE1
    CACHE1 --> GT
    
    GT --> SPLIT
    SPLIT --> FSEM
    SPLIT --> FCTX
    SPLIT --> FNAV
    
    FSEM --> FEAT
    FCTX --> FEAT
    FNAV --> FEAT
    
    FEAT --> XGB
    XGB --> MODEL
    
    MODEL --> METRICS
    METRICS --> VIZ
    
    VIZ --> R1
    VIZ --> R2
    VIZ --> R3
    VIZ --> R4
    VIZ --> R5
```

**Export:**
1. Paste into Mermaid Live
2. Click "Actions" → "Download PNG"
3. Save as `pipeline_complete.png` in this directory

---

### 3. Export Diagram 2: Research Methodology

**Source:** Find **Diagram 2: Research Methodology Flow** in `docs/PIPELINE_DIAGRAMS.md`

**Copy and paste the mermaid code, then:**
1. Export as PNG
2. Save as `methodology_flow.png`

---

### 4. Export Diagram 5: Statistical Analysis

**Source:** Find **Diagram 5: Cross-Domain Statistical Analysis** in `docs/PIPELINE_DIAGRAMS.md`

**Copy and paste the mermaid code, then:**
1. Export as PNG
2. Save as `statistical_analysis.png`

---

## Done!

You now have the 3 essential diagrams for your thesis Results chapter.

**Optional:** Export Diagrams 3, 4, 6, 7, 8 using the same process if needed.

---

## Alternative: Screenshot from Cursor

1. Open `docs/PIPELINE_DIAGRAMS.md` in Cursor
2. Click preview (diagrams render automatically)
3. Press `Cmd+Shift+4` (macOS) to screenshot
4. Drag to select diagram area
5. Save to this directory

---

**Estimated time:** 15-20 minutes for all 3 priority diagrams
