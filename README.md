# Hybrid Hallucination Prediction Framework (HHPF)

A proactive hallucination prediction system for Large Language Models (LLMs) that analyzes internal model signals and prompt characteristics to calculate real-time hallucination risk scores.

## Overview

This Master's thesis project shifts from reactive hallucination detection to proactive prediction by analyzing:
- **Epistemic Uncertainty**: Semantic Entropy and Semantic Energy from model internals
- **Contextual Features**: Knowledge Popularity and Prompt Complexity
- **Domain Patterns**: Cross-domain hallucination behavior across Medicine, Math, Finance, IS, and Psychology

## Research Question

**"Which features in the user query and the required knowledge structure are most significantly correlated with the occurrence of hallucinations?"**

## Architecture

```
HHPF Pipeline:
1. Data Preparation → 2. Inference (Llama-3) → 3. Feature Extraction → 4. XGBoost Classifier → 5. Evaluation
```

**Key Components:**
- **Inference Layer**: Llama-3 (8B/70B) for response generation and internal signal extraction
- **Signal Extraction**: Semantic entropy (NLI clustering), semantic energy (logit analysis), knowledge popularity, prompt complexity
- **Classifier**: XGBoost binary classifier (hallucination vs. faithful)
- **Evaluation**: AUROC, Accuracy-Rejection Curve (ARC), Expected Calibration Error (ECE), Feature Importance

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- macOS (M1/M2) or Linux with pip
- API key from [Together AI](https://api.together.xyz/) or [Groq](https://console.groq.com/)

### 2. Installation

```bash
# Clone or navigate to the project directory
cd HHPF

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. API Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# - Get Together AI key: https://api.together.xyz/settings/api-keys
# - Get Groq key: https://console.groq.com/keys
```

**Recommended API Provider:** Together AI
- Cost-effective: ~$0.20/1M tokens (Llama-3-8B), ~$0.88/1M tokens (Llama-3-70B)
- Supports logprobs extraction (required for semantic energy)
- Good rate limits for research projects

### 4. Verify Installation

```bash
# Test that all imports work
python -c "import torch, transformers, xgboost, together; print('✓ All dependencies installed')"

# Verify spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ spaCy model loaded')"
```

## Project Structure

```
HHPF/
├── data/
│   ├── raw/                    # Raw dataset CSVs (place your datasets here)
│   ├── processed/              # Processed datasets with labels
│   └── features/               # Extracted feature matrices
├── src/
│   ├── data_preparation/       # Dataset loaders and ground truth extraction
│   ├── inference/              # Llama-3 API client and response generation
│   ├── features/               # Feature extraction (uncertainty + context)
│   ├── classifier/             # XGBoost training and prediction
│   └── evaluation/             # Metrics and visualization
├── notebooks/                  # Jupyter notebooks for analysis
├── configs/                    # YAML configuration files
├── outputs/
│   ├── models/                 # Trained models
│   ├── figures/                # Publication-ready plots
│   └── results/                # Metrics and analysis
├── requirements.txt
├── .env                        # API keys (not in git)
└── README.md
```

## Datasets

Place your raw datasets in `data/raw/`:
- **Medicine**: Med-HALT (clinical reasoning)
- **Math**: GSM8K/Socratic (step-by-step logic)
- **Finance**: FinanceBench (numerical accuracy)
- **IS/Agents**: HalluMix (autonomous agent scenarios)
- **Psychology/General**: TruthfulQA (cognitive biases)

## Usage

### Quick Start (Full Pipeline)

```bash
# 1. Prepare datasets
python -m src.data_preparation.process_datasets

# 2. Run inference and extract features
python -m src.inference.generate_responses
python -m src.features.extract_all_features

# 3. Train XGBoost classifier
python -m src.classifier.train_model

# 4. Evaluate and generate plots
python -m src.evaluation.evaluate_model
```

### Step-by-Step (Recommended for Development)

```bash
# Start with one domain (Math) as proof-of-concept
python -m src.data_preparation.process_datasets --domain math

# Generate responses with Llama-3-8B
python -m src.inference.generate_responses --dataset math --model llama-3-8b

# Extract features
python -m src.features.extract_all_features --dataset math

# Train initial model
python -m src.classifier.train_model --dataset math

# View results in Jupyter notebook
jupyter notebook notebooks/03_model_training.ipynb
```

## Key Metrics

- **AUROC**: Primary performance metric (target: >0.75)
- **Accuracy-Rejection Curve**: Model's selective prediction ability
- **Expected Calibration Error**: Probability calibration quality
- **Feature Importance**: Statistical correlation with hallucinations

## Cost Estimation

**Using Together AI with Llama-3-8B:**
- ~5,000 prompts × 10 samples × 500 tokens = 25M tokens
- Cost: ~$5-7 for full dataset
- Time: 2-4 hours with parallel processing

**Using Llama-3-70B (if needed):**
- Same volume: ~$20-25
- Better performance on complex domains

## Development Workflow

1. **Proof-of-Concept**: Start with GSM8K (Math) - simplest ground truth
2. **Incremental Features**: Add features one at a time, measure impact
3. **Scale Up**: Expand to all 5 domains once pipeline is validated
4. **Optimization**: Hyperparameter tuning and ablation studies
5. **Analysis**: Generate figures and statistical tests for thesis

## Troubleshooting

**API Rate Limits:**
- Implement delays between requests (built into `llama_client.py`)
- Cache all responses to avoid redundant calls

**Memory Issues on M1:**
- Process datasets in batches
- Use MPS (Metal Performance Shaders) for PyTorch: `device = torch.device("mps")`
- Reduce `NUM_STOCHASTIC_SAMPLES` if needed

**spaCy Model Errors:**
```bash
# Reinstall spaCy model
python -m spacy download en_core_web_sm --force
```

## Citation

If you use this framework in your research, please cite:

```
[Your thesis citation will go here]
```

## License

[Specify license - typically MIT or academic use only]

## Contact

[Your contact information]
