"""
Inspect dataset structure and columns.

Helps identify the correct column names for configs/datasets.yaml
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_dataset(filepath: Path):
    """Inspect a single dataset file."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {filepath.name}")
    logger.info(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath)
        
        logger.info(f"Rows: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        logger.info(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        # Column types
        logger.info(f"\nColumn types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.info(f"\nMissing values:")
            for col, count in missing[missing > 0].items():
                logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sample values for key columns
        logger.info(f"\nSample values:")
        for col in df.columns[:5]:  # First 5 columns
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            if isinstance(sample, str) and len(sample) > 100:
                sample = sample[:100] + "..."
            logger.info(f"  {col}: {sample}")
        
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")


def main():
    """Inspect all datasets."""
    raw_dir = Path("data/raw")
    
    datasets = [
        "gsm8k.csv",
        "med_halt.csv",
        "financebench.csv",
        "hallumix.csv",
        "truthfulqa.csv"
    ]
    
    logger.info("="*60)
    logger.info("HHPF Dataset Inspector")
    logger.info("="*60)
    
    for dataset_file in datasets:
        filepath = raw_dir / dataset_file
        inspect_dataset(filepath)
    
    logger.info("\n" + "="*60)
    logger.info("Inspection Complete")
    logger.info("="*60)
    logger.info("\nUse this information to update configs/datasets.yaml")
    logger.info("Make sure 'prompt_column' and 'answer_column' match actual column names")


if __name__ == "__main__":
    main()
