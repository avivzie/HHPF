# HHPF Project Flow Diagrams

**Purpose:** Visual representation of the complete HHPF research workflow for thesis Results chapter  
**Date:** February 14, 2026  
**Status:** Thesis-ready diagrams

---

## Diagram 1: Complete HHPF Technical Pipeline

This diagram shows the complete technical execution pipeline from raw data to per-domain results.

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

**Key Points:**
- **5 diverse domains** enter the pipeline independently
- **Caching** at response and feature stages enables efficient reprocessing
- **3 feature categories** (semantic, contextual, naive) extracted for each sample
- **Per-domain training** ensures no cross-domain contamination
- **Each domain produces** independent model and metrics

---

## Diagram 2: Research Methodology Flow (Phase A → Phase B)

This diagram shows how per-domain results are analyzed to answer the three research questions.

```mermaid
flowchart TD
    subgraph domains [Per-Domain Results from Pipeline]
        M1[Math<br/>AUROC 0.797]
        M2[IS Agents<br/>AUROC 0.703]
        M3[Psychology<br/>AUROC 0.671]
        M4[Finance<br/>AUROC 0.632]
        M5[Medicine<br/>AUROC 0.619]
    end
    
    subgraph phaseA [Phase A: Per-Domain Ablation]
        direction TB
        ABL[Run 5 Feature Subsets<br/>Per Domain]
        FS1[Naive-Only]
        FS2[Semantic-Only]
        FS3[Context-Only]
        FS4[Semantic+Context]
        FS5[Full Model]
        
        ABL --> FS1
        ABL --> FS2
        ABL --> FS3
        ABL --> FS4
        ABL --> FS5
    end
    
    MODELS[25 Models Trained<br/>5 domains × 5 subsets<br/>Same XGBoost config]
    
    subgraph phaseB [Phase B: Cross-Domain Analysis]
        AGG[Aggregate Results<br/>Mean ± Std across domains]
        STAT[Statistical Tests]
        TTEST1[Paired t-test: RQ1]
        TTEST2[Paired t-test: RQ2]
        CHI[Chi-square: RQ3a]
        CV[Variance Analysis: RQ3c]
    end
    
    subgraph results [Research Question Answers]
        RQ1[RQ1: Hybrid > Baseline<br/>p=0.087, d=1.007<br/>Trend-level support]
        RQ2[RQ2: Semantic > Naive<br/>p=0.262, d=0.312<br/>Not significant]
        RQ3[RQ3: Domain Variance<br/>p<0.001<br/>Strongly supported]
    end
    
    M1 --> ABL
    M2 --> ABL
    M3 --> ABL
    M4 --> ABL
    M5 --> ABL
    
    FS1 --> MODELS
    FS2 --> MODELS
    FS3 --> MODELS
    FS4 --> MODELS
    FS5 --> MODELS
    
    MODELS --> AGG
    AGG --> STAT
    
    STAT --> TTEST1
    STAT --> TTEST2
    STAT --> CHI
    STAT --> CV
    
    TTEST1 --> RQ1
    TTEST2 --> RQ2
    CHI --> RQ3
    CV --> RQ3
```

**Key Points:**
- **No cross-domain training** - each domain analyzed independently in Phase A
- **5 feature subsets** tested per domain for fair comparison
- **Aggregation** in Phase B combines per-domain results statistically
- **Proper statistical tests** (paired t-tests, chi-square) answer research questions
- **Transparent reporting** of p-values and effect sizes

---

## Diagram 3: Feature Engineering Architecture

This diagram shows the three categories of features extracted from LLM responses.

```mermaid
flowchart LR
    subgraph llm [LLM Response]
        PRIMARY[Primary Response]
        STOCH[5 Stochastic Samples<br/>temp 0.7-1.0]
        LOGPROBS[Token LogProbs]
    end
    
    subgraph semantic [Semantic Uncertainty]
        SE[Semantic Entropy<br/>NLI-based clustering]
        SENG[Semantic Energy<br/>Logit distribution]
        CLUST[Cluster Statistics<br/>Count, size, diversity]
    end
    
    subgraph contextual [Contextual Features]
        ENT[Entity Features<br/>Rarity, type, count]
        LEX[Lexical Features<br/>Token stats, diversity]
        SYN[Syntactic Features<br/>Parse depth, complexity]
        QTYPE[Question Type<br/>What, why, how, etc.]
    end
    
    subgraph naive [Naive Confidence]
        MAXP[Max Probability]
        PERP[Perplexity]
        MEANLOG[Mean/Min LogProb]
    end
    
    FEAT[Feature Vector<br/>41-48 dimensions]
    
    PRIMARY --> SE
    STOCH --> SE
    STOCH --> SENG
    STOCH --> CLUST
    
    PRIMARY --> ENT
    PRIMARY --> LEX
    PRIMARY --> SYN
    PRIMARY --> QTYPE
    
    LOGPROBS --> MAXP
    LOGPROBS --> PERP
    LOGPROBS --> MEANLOG
    
    SE --> FEAT
    SENG --> FEAT
    CLUST --> FEAT
    ENT --> FEAT
    LEX --> FEAT
    SYN --> FEAT
    QTYPE --> FEAT
    MAXP --> FEAT
    PERP --> FEAT
    MEANLOG --> FEAT
```

**Key Points:**
- **Semantic features** require multiple stochastic samples for uncertainty quantification
- **Contextual features** analyze the question and response content
- **Naive features** use model's internal confidence signals
- **All features** combined into unified representation

---

## Diagram 4: Ablation Study Design

This diagram illustrates the per-domain ablation methodology used for RQ1 and RQ2.

```mermaid
flowchart TD
    subgraph domain [Single Domain Example: Math]
        DATA[Feature Matrix<br/>542 samples<br/>41 features]
        TRAIN[Train: 434 samples]
        TEST[Test: 108 samples]
    end
    
    subgraph subsets [5 Feature Subsets]
        S1[Subset 1: Naive-Only<br/>4 features]
        S2[Subset 2: Semantic-Only<br/>3 features]
        S3[Subset 3: Context-Only<br/>21 features]
        S4[Subset 4: Semantic+Context<br/>24 features]
        S5[Subset 5: Full<br/>41 features]
    end
    
    subgraph models [Train 5 Models]
        XGB1[XGBoost<br/>Same config]
        XGB2[XGBoost<br/>Same config]
        XGB3[XGBoost<br/>Same config]
        XGB4[XGBoost<br/>Same config]
        XGB5[XGBoost<br/>Same config]
    end
    
    subgraph eval [Evaluate on Test Set]
        E1[AUROC: 0.500]
        E2[AUROC: 0.722]
        E3[AUROC: 0.570]
        E4[AUROC: 0.762]
        E5[AUROC: 0.797]
    end
    
    RESULT[Per-Domain<br/>Ablation Results]
    
    DATA --> TRAIN
    DATA --> TEST
    
    TRAIN --> S1
    TRAIN --> S2
    TRAIN --> S3
    TRAIN --> S4
    TRAIN --> S5
    
    S1 --> XGB1
    S2 --> XGB2
    S3 --> XGB3
    S4 --> XGB4
    S5 --> XGB5
    
    XGB1 --> E1
    XGB2 --> E2
    XGB3 --> E3
    XGB4 --> E4
    XGB5 --> E5
    
    TEST --> E1
    TEST --> E2
    TEST --> E3
    TEST --> E4
    TEST --> E5
    
    E1 --> RESULT
    E2 --> RESULT
    E3 --> RESULT
    E4 --> RESULT
    E5 --> RESULT
```

**Key Points:**
- **Same train/test split** used for all feature subsets (fair comparison)
- **Identical XGBoost config** for all models (eliminates confounding)
- **5 independent models** trained per domain
- **Repeated for all 5 domains** = 25 total models

---

## Diagram 5: Cross-Domain Statistical Analysis

This diagram shows how per-domain ablation results are combined to answer research questions.

```mermaid
flowchart TD
    subgraph ablation [Per-Domain Ablation Results]
        direction LR
        A1[Math<br/>5 subsets]
        A2[IS Agents<br/>5 subsets]
        A3[Psychology<br/>5 subsets]
        A4[Medicine<br/>5 subsets]
        A5[Finance<br/>5 subsets]
    end
    
    subgraph aggregate [Aggregation]
        AGG[Calculate Statistics<br/>Mean ± Std<br/>Min, Max]
    end
    
    subgraph rq1 [RQ1: Hybrid vs Baseline]
        COMP1[Compare:<br/>Semantic+Context<br/>vs Naive-Only]
        TEST1[Paired t-test<br/>n=5 domain pairs]
        RES1[Result:<br/>p=0.087<br/>Cohen's d=1.007<br/>Trend-level]
    end
    
    subgraph rq2 [RQ2: Semantic vs Naive]
        COMP2[Compare:<br/>Semantic-Only<br/>vs Naive-Only]
        TEST2[Paired t-test<br/>one-tailed, n=5]
        RES2[Result:<br/>p=0.262<br/>Cohen's d=0.312<br/>Not significant]
    end
    
    subgraph rq3 [RQ3: Cross-Domain Variance]
        COMP3A[Hallucination Rates<br/>5 domains]
        COMP3B[AUROC Variance<br/>Range: 0.619-0.797]
        COMP3C[Feature Importance<br/>CV analysis]
        TEST3A[Chi-square test]
        TEST3C[CV > 0.3 threshold]
        RES3[Result:<br/>p<0.001<br/>63% high-variance<br/>Strongly supported]
    end
    
    A1 --> AGG
    A2 --> AGG
    A3 --> AGG
    A4 --> AGG
    A5 --> AGG
    
    AGG --> COMP1
    AGG --> COMP2
    AGG --> COMP3A
    AGG --> COMP3B
    AGG --> COMP3C
    
    COMP1 --> TEST1
    TEST1 --> RES1
    
    COMP2 --> TEST2
    TEST2 --> RES2
    
    COMP3A --> TEST3A
    COMP3C --> TEST3C
    TEST3A --> RES3
    COMP3B --> RES3
    TEST3C --> RES3
```

**Key Points:**
- **No cross-domain training** - results aggregated statistically, not by pooling data
- **Paired tests** for RQ1/RQ2 (within-domain comparisons across 5 domains)
- **Multiple analyses** for RQ3 (hallucination rates, AUROC variance, feature importance)
- **Transparent reporting** of p-values and effect sizes

---

## Diagram 6: High-Level Conceptual Overview

This simplified diagram provides a conceptual understanding of the complete research workflow.

```mermaid
flowchart TD
    INPUT[5 Diverse Domains<br/>Math, IS, Psych, Med, Finance<br/>n=2,542 samples]
    
    PIPELINE[HHPF Pipeline<br/>Inference → Labeling → Features]
    
    subgraph hybrid [Hybrid Feature Framework]
        SEM[Semantic Uncertainty<br/>Entropy, Energy]
        CTX[Contextual Signals<br/>Complexity, Rarity]
        NAV[Naive Confidence<br/>Probabilities]
    end
    
    PERDOM[Per-Domain Analysis<br/>5 feature subsets × 5 domains<br/>25 models trained]
    
    CROSSDOM[Cross-Domain Comparison<br/>Paired statistics<br/>No data leakage]
    
    subgraph findings [Research Findings]
        F1[RQ1: Hybrid features<br/>show large effect<br/>Cohen's d=1.007]
        F2[RQ2: Semantic alone<br/>insufficient<br/>p=0.262]
        F3[RQ3: Domains differ<br/>significantly<br/>p<0.001]
    end
    
    INPUT --> PIPELINE
    PIPELINE --> SEM
    PIPELINE --> CTX
    PIPELINE --> NAV
    
    SEM --> PERDOM
    CTX --> PERDOM
    NAV --> PERDOM
    
    PERDOM --> CROSSDOM
    
    CROSSDOM --> F1
    CROSSDOM --> F2
    CROSSDOM --> F3
```

**Key Message:** The HHPF framework uses hybrid features analyzed per-domain with cross-domain statistical comparison to reveal domain-dependent hallucination patterns.

---

## Diagram 7: Complete End-to-End Research Flow

This comprehensive diagram shows the entire research workflow from data collection to thesis conclusions.

```mermaid
flowchart TB
    subgraph data [Data Collection]
        RAW[Raw Datasets<br/>5 domains<br/>2,542 samples total]
    end
    
    subgraph pipeline [Technical Pipeline]
        PROC[Data Processing]
        INF[LLM Inference<br/>5 samples per prompt]
        LABEL[Ground Truth<br/>Domain-specific methods]
        SPLIT[80/20 Stratified Split]
        FEXT[Feature Extraction<br/>41-48 features]
        TRAIN[XGBoost Training<br/>Hyperparameter tuning]
        EVAL[Evaluation<br/>AUROC, Accuracy, ECE]
    end
    
    DOMAIN_RES[5 Per-Domain Results<br/>Models + Metrics + Figures]
    
    subgraph phase_a [Phase A: Per-Domain Ablation]
        ABL_SCRIPT[per_domain_ablation.py]
        ABL_RUN[Run 5 feature subsets<br/>for each domain]
        ABL_OUT[25 ablation results<br/>5 feature importance files]
    end
    
    subgraph phase_b [Phase B: Statistical Analysis]
        AGG_SCRIPT[aggregate_ablation_results.py]
        STAT_SCRIPT[statistical_tests.py]
        VIZ_SCRIPT[generate_thesis_figures.py]
        
        AGG_RES[Aggregated Results<br/>Mean ± Std]
        STATS[Statistical Tests<br/>t-tests, chi-square]
        FIGS[10 Publication Figures<br/>PDF + PNG]
    end
    
    subgraph rq [Research Questions Answered]
        RQ1_ANS[RQ1: Hybrid features<br/>Large effect, trend-level<br/>p=0.087]
        RQ2_ANS[RQ2: Semantic insufficient<br/>Small effect<br/>p=0.262]
        RQ3_ANS[RQ3: Domains differ<br/>Highly significant<br/>p<0.001]
    end
    
    THESIS[Thesis Results Section<br/>Complete with figures,<br/>tables, and statistical<br/>evidence]
    
    RAW --> PROC
    PROC --> INF
    INF --> LABEL
    LABEL --> SPLIT
    SPLIT --> FEXT
    FEXT --> TRAIN
    TRAIN --> EVAL
    EVAL --> DOMAIN_RES
    
    DOMAIN_RES --> ABL_SCRIPT
    ABL_SCRIPT --> ABL_RUN
    ABL_RUN --> ABL_OUT
    
    ABL_OUT --> AGG_SCRIPT
    AGG_SCRIPT --> AGG_RES
    
    AGG_RES --> STAT_SCRIPT
    STAT_SCRIPT --> STATS
    
    STATS --> RQ1_ANS
    STATS --> RQ2_ANS
    STATS --> RQ3_ANS
    
    ABL_OUT --> VIZ_SCRIPT
    AGG_RES --> VIZ_SCRIPT
    VIZ_SCRIPT --> FIGS
    
    RQ1_ANS --> THESIS
    RQ2_ANS --> THESIS
    RQ3_ANS --> THESIS
    FIGS --> THESIS
```

**Complete Research Loop:**
1. Data Collection (5 domains, 2,542 samples)
2. Technical Pipeline (inference → features → training → evaluation)
3. Per-Domain Results (5 independent models)
4. Phase A: Per-Domain Ablation (25 models across feature subsets)
5. Phase B: Cross-Domain Analysis (statistical aggregation)
6. Research Questions Answered (with p-values and effect sizes)
7. Thesis-Ready Deliverables (figures, tables, documentation)

---

## Diagram 8: Data Flow Architecture

This diagram shows how data flows through the system and where results are stored.

```mermaid
flowchart LR
    subgraph raw [data/raw/]
        R1[GSM8K]
        R2[HalluMix]
        R3[TruthfulQA]
        R4[Med-HALT]
        R5[TAT-QA]
    end
    
    subgraph processed [data/processed/]
        P1[math_processed.csv]
        P2[is_agents_processed.csv]
        P3[psychology_processed.csv]
        P4[medicine_processed.csv]
        P5[finance_processed.csv]
    end
    
    subgraph features_data [data/features/]
        F1[math_features.csv<br/>542×41]
        F2[is_agents_features.csv<br/>500×41]
        F3[psychology_features.csv<br/>500×41]
        F4[medicine_features.csv<br/>500×48]
        F5[finance_features.csv<br/>500×41]
        RESP[(Response Cache<br/>~2,500 .pkl files)]
    end
    
    subgraph outputs_dir [outputs/]
        MODELS[models/<br/>5 XGBoost .pkl]
        RESULTS[results/<br/>5 metrics JSON]
        FIGURES[figures/<br/>25 visualizations]
        ABLATION[ablation/<br/>10 CSV files]
        RQ[research_questions/<br/>Results + Figures]
    end
    
    R1 --> P1
    R2 --> P2
    R3 --> P3
    R4 --> P4
    R5 --> P5
    
    P1 --> F1
    P2 --> F2
    P3 --> F3
    P4 --> F4
    P5 --> F5
    
    P1 --> RESP
    P2 --> RESP
    P3 --> RESP
    P4 --> RESP
    P5 --> RESP
    
    F1 --> MODELS
    F2 --> MODELS
    F3 --> MODELS
    F4 --> MODELS
    F5 --> MODELS
    
    MODELS --> RESULTS
    MODELS --> FIGURES
    MODELS --> ABLATION
    
    ABLATION --> RQ
    RESULTS --> RQ
```

**Storage Organization:**
- **`data/`** - Raw datasets, processed CSVs, feature matrices, response cache
- **`outputs/models/`** - Trained XGBoost models (one per domain)
- **`outputs/results/`** - Per-domain metrics JSON files
- **`outputs/figures/`** - Per-domain visualizations (ROC, ARC, calibration, etc.)
- **`outputs/ablation/`** - Ablation study results and feature importance
- **`outputs/research_questions/`** - Final RQ analysis, statistical tests, publication figures

---

## Usage Instructions for Thesis

### Including Diagrams in Results Chapter

**For LaTeX:**
1. Export diagrams as PDF or PNG (high resolution)
2. Include using `\includegraphics{figures/pipeline_complete.pdf}`
3. Add caption explaining the workflow

**For Word/Google Docs:**
1. Export diagrams as PNG (300 DPI)
2. Insert as inline images
3. Add figure caption below

**Recommended Placement:**

**Section 4.1 - Experimental Setup:**
- Use **Diagram 1** (Complete Pipeline) to show technical workflow
- Use **Diagram 3** (Feature Engineering) to explain feature extraction

**Section 4.2 - Analysis Methodology:**
- Use **Diagram 2** (Research Methodology) to show Phase A → Phase B
- Use **Diagram 4** (Ablation Design) to illustrate per-domain approach
- Use **Diagram 5** (Statistical Analysis) to show how RQ are answered

**Section 4.0 - Overview (Optional):**
- Use **Diagram 6** (Conceptual Overview) for high-level introduction

**Appendix - Technical Details:**
- Use **Diagram 8** (Data Flow) to document system architecture

### Exporting Diagrams

**Method 1: Mermaid Live Editor**
1. Copy mermaid code to https://mermaid.live
2. Export as PNG (300 DPI) or SVG
3. Convert SVG to PDF if needed

**Method 2: Cursor Preview**
1. View diagram in Cursor markdown preview
2. Take screenshot (high resolution)
3. Crop and save

**Method 3: Mermaid CLI (Recommended)**
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export diagrams
mmdc -i docs/PIPELINE_DIAGRAMS.md -o outputs/research_questions/figures/pipeline_complete.png -w 2400
```

---

## Diagram Summary

| Diagram | Purpose | Best For | Complexity |
|---------|---------|----------|------------|
| 1. Complete Pipeline | Technical workflow | Methods/Results 4.1 | Medium |
| 2. Research Methodology | Phase A → Phase B | Results 4.2 | Medium |
| 3. Feature Engineering | Feature architecture | Methods | Low |
| 4. Ablation Design | Per-domain approach | Methods detail | Low |
| 5. Statistical Analysis | RQ answering process | Results 4.2 | Medium |
| 6. Conceptual Overview | High-level summary | Introduction/Overview | Low |
| 7. End-to-End Flow | Complete research loop | Appendix | High |
| 8. Data Flow | System architecture | Technical appendix | Medium |

**Recommended for Results Chapter:**
- **Must include:** Diagrams 1, 2, 5 (show complete story)
- **Optional but helpful:** Diagrams 3, 4 (explain details)
- **For appendix:** Diagrams 7, 8 (technical documentation)

---

## Key Messages Conveyed

1. **Rigorous Methodology:** Per-domain analysis prevents data leakage
2. **Hybrid Approach:** Combines semantic, contextual, and naive features
3. **Statistical Rigor:** Proper paired tests with p-values and effect sizes
4. **Domain Dependency:** Hallucination patterns vary significantly (RQ3)
5. **Comprehensive Analysis:** 30 models trained, 10 figures generated

---

**Diagrams Created:** February 14, 2026  
**Status:** Thesis-ready  
**Format:** Mermaid (can export to PNG/PDF/SVG)  
**Location:** `docs/PIPELINE_DIAGRAMS.md`
