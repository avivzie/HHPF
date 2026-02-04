"""
Dataset preparation script for HHPF.

Consolidates existing datasets and downloads missing ones from HuggingFace.
"""

import pandas as pd
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def consolidate_gsm8k():
    """Consolidate GSM8K train/test files into single CSV."""
    logger.info("Consolidating GSM8K dataset...")
    
    raw_dir = Path("data/raw")
    gsm8k_dir = raw_dir / "GSM8K"
    
    if not gsm8k_dir.exists():
        logger.error("GSM8K directory not found")
        return False
    
    # Read all GSM8K files
    files = {
        'main_train': gsm8k_dir / "main_train.csv",
        'main_test': gsm8k_dir / "main_test.csv",
    }
    
    dfs = []
    
    for name, filepath in files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['source'] = name
            dfs.append(df)
            logger.info(f"  Loaded {name}: {len(df)} samples")
        else:
            logger.warning(f"  File not found: {filepath}")
    
    if not dfs:
        logger.error("No GSM8K files found")
        return False
    
    # Combine
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save consolidated file
    output_path = raw_dir / "gsm8k.csv"
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Consolidated GSM8K saved to {output_path}")
    logger.info(f"  Total samples: {len(combined_df)}")
    
    return True


def download_hallumix():
    """Download HalluMix dataset from HuggingFace."""
    logger.info("Downloading HalluMix dataset...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return False
    
    try:
        # Load dataset
        dataset = load_dataset("quotientai/HalluMix")
        
        # Convert to pandas
        # HalluMix may have multiple splits, combine them
        dfs = []
        for split_name in dataset.keys():
            df = dataset[split_name].to_pandas()
            df['split'] = split_name
            dfs.append(df)
            logger.info(f"  Loaded {split_name}: {len(df)} samples")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save
        output_path = Path("data/raw/hallumix.csv")
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ HalluMix saved to {output_path}")
        logger.info(f"  Total samples: {len(combined_df)}")
        logger.info(f"  Columns: {list(combined_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download HalluMix: {e}")
        return False


def download_med_halt():
    """Download Med-HALT dataset from HuggingFace."""
    logger.info("Downloading Med-HALT dataset...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return False
    
    try:
        # Med-HALT has multiple configs - use reasoning configs which are most relevant
        configs_to_use = ['reasoning_fake', 'reasoning_nota', 'reasoning_FCT']
        
        all_dfs = []
        
        for config_name in configs_to_use:
            logger.info(f"  Downloading config: {config_name}")
            dataset = load_dataset("openlifescienceai/Med-HALT", config_name)
            
            # Convert to pandas
            for split_name in dataset.keys():
                df = dataset[split_name].to_pandas()
                df['split'] = split_name
                df['config'] = config_name
                all_dfs.append(df)
                logger.info(f"    Loaded {config_name}/{split_name}: {len(df)} samples")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save
        output_path = Path("data/raw/med_halt.csv")
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Med-HALT saved to {output_path}")
        logger.info(f"  Total samples: {len(combined_df)}")
        logger.info(f"  Columns: {list(combined_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Med-HALT: {e}")
        return False


def check_existing_datasets():
    """Check which datasets are already present."""
    logger.info("Checking existing datasets...")
    
    raw_dir = Path("data/raw")
    expected_files = {
        'gsm8k.csv': 'Math (GSM8K)',
        'med_halt.csv': 'Medicine (Med-HALT)',
        'financebench_sample_150.csv': 'Finance (FinanceBench)',
        'hallumix.csv': 'IS/Agents (HalluMix)',
        'TruthfulQA.csv': 'Psychology (TruthfulQA)'
    }
    
    found = []
    missing = []
    
    for filename, description in expected_files.items():
        filepath = raw_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            found.append((filename, description, len(df)))
            logger.info(f"  ✓ {filename}: {len(df)} samples")
        else:
            missing.append((filename, description))
            logger.info(f"  ✗ {filename}: NOT FOUND")
    
    return found, missing


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HHPF datasets")
    parser.add_argument(
        '--consolidate-gsm8k',
        action='store_true',
        help='Consolidate GSM8K train/test files'
    )
    parser.add_argument(
        '--download-hallumix',
        action='store_true',
        help='Download HalluMix from HuggingFace'
    )
    parser.add_argument(
        '--download-med-halt',
        action='store_true',
        help='Download Med-HALT from HuggingFace'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preparation steps'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check which datasets are present'
    )
    
    args = parser.parse_args()
    
    # If no specific action, show help
    if not any([args.consolidate_gsm8k, args.download_hallumix, 
                args.download_med_halt, args.all, args.check]):
        parser.print_help()
        sys.exit(0)
    
    logger.info("="*60)
    logger.info("HHPF Dataset Preparation")
    logger.info("="*60)
    
    # Check existing datasets first
    if args.check or args.all:
        logger.info("\n" + "="*60)
        logger.info("Checking Existing Datasets")
        logger.info("="*60)
        found, missing = check_existing_datasets()
    
    # Consolidate GSM8K
    if args.consolidate_gsm8k or args.all:
        logger.info("\n" + "="*60)
        logger.info("Step 1: Consolidating GSM8K")
        logger.info("="*60)
        consolidate_gsm8k()
    
    # Download HalluMix
    if args.download_hallumix or args.all:
        logger.info("\n" + "="*60)
        logger.info("Step 2: Downloading HalluMix")
        logger.info("="*60)
        download_hallumix()
    
    # Download Med-HALT
    if args.download_med_halt or args.all:
        logger.info("\n" + "="*60)
        logger.info("Step 3: Downloading Med-HALT")
        logger.info("="*60)
        download_med_halt()
    
    # Final check
    logger.info("\n" + "="*60)
    logger.info("Final Dataset Status")
    logger.info("="*60)
    found, missing = check_existing_datasets()
    
    if missing:
        logger.info(f"\n⚠️  Missing {len(missing)} datasets:")
        for filename, description in missing:
            logger.info(f"  - {filename} ({description})")
        logger.info("\nTo add missing datasets:")
        logger.info("  - Place CSV files in data/raw/")
        logger.info("  - Or run with appropriate --download-* flags")
    else:
        logger.info("\n✓ All datasets present!")
    
    logger.info("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
