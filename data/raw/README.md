# Raw Datasets

Place your raw dataset CSV files in this directory.

## Expected Datasets

### 1. Med-HALT (Medicine)
- **File**: `med_halt.csv`
- **Domain**: Medicine
- **Description**: Clinical reasoning questions with ground truth medical answers
- **Source**: [Link to dataset source]

### 2. GSM8K (Math)
- **File**: `gsm8k.csv`
- **Domain**: Math
- **Description**: Grade school math word problems with numerical answers
- **Source**: https://github.com/openai/grade-school-math

### 3. FinanceBench (Finance)
- **File**: `financebench.csv`
- **Domain**: Finance
- **Description**: Financial Q&A with numerical and factual answers
- **Source**: [Link to dataset source]

### 4. HalluMix (IS/Agents)
- **File**: `hallumix.csv`
- **Domain**: Information Systems
- **Description**: Autonomous agent scenarios requiring factual accuracy
- **Source**: [Link to dataset source]

### 5. TruthfulQA (Psychology/General)
- **File**: `truthfulqa.csv`
- **Domain**: Psychology
- **Description**: Questions testing cognitive biases and common misconceptions
- **Source**: https://github.com/sylinrl/TruthfulQA

## CSV Format Requirements

Each dataset should have at minimum:
- **Prompt column**: The question or input text
- **Answer column**: The ground truth answer or reference
- **ID column** (optional): Unique identifier for each example

The column names will be mapped in `configs/datasets.yaml`.

## Download Instructions

1. Download datasets from their respective sources
2. Convert to CSV format if necessary
3. Place in this directory with the filenames listed above
4. Update `configs/datasets.yaml` if column names differ

## Notes

- Raw datasets are .gitignored to avoid committing potentially large files
- Ensure you have the right to use these datasets for research
- Document any preprocessing done before placing files here
