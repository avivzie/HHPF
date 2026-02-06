"""
Main script for processing datasets.

Loads raw datasets, formats prompts, and prepares for inference.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

from src.data_preparation.dataset_loaders import get_loader
from src.data_preparation.prompt_formatter import PromptFormatter
from src.utils import load_config, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_dataset(domain: str, output_dir: str = "data/processed", limit: int = None):
    """
    Process a single dataset.
    
    Args:
        domain: Domain name (medicine, math, finance, is_agents, psychology)
        output_dir: Output directory for processed data
        limit: Limit number of samples (applied BEFORE train/test split)
    """
    logger.info(f"Processing {domain} dataset...")
    
    # Load configuration
    config = load_config("datasets")
    global_config = config['global']
    
    # Load dataset
    loader = get_loader(domain)
    df = loader.load_dataset(domain)
    
    # Extract prompts and ground truth
    df = loader.get_prompt_and_answer(domain, df)
    
    # Validate
    loader.validate_dataset(df, domain)
    
    # Preprocess (domain-specific)
    df = loader.preprocess(df)
    
    # Apply limit BEFORE train/test split to maintain ratio
    if limit:
        # Shuffle before limiting to avoid sorted dataset bias
        random_seed = global_config.get('random_seed', 42)
        df = df.sample(n=min(limit, len(df)), random_state=random_seed).reset_index(drop=True)
        logger.info(f"Limited to {limit} samples (randomly sampled) before splitting")
    
    # Format prompts
    formatter = PromptFormatter(template_style="simple")
    dataset_config = config['datasets'][domain]
    domain_name = dataset_config['domain']
    
    df['formatted_prompt'] = df['prompt'].apply(
        lambda x: formatter.format_prompt(x, domain_name)
    )
    
    # Create train/test split
    train_ratio = global_config.get('train_test_split', 0.8)
    random_seed = global_config.get('random_seed', 42)
    
    # Stratification for domains with heterogeneous answer types
    stratify_column = None
    if domain == 'medicine':
        # Stratify by "None of the above" vs specific answers
        # This ensures proportional representation in train/test sets
        df['_stratify_label'] = df['ground_truth'].str.lower().isin([
            'none of the above', 'none', 'no correct answer'
        ]).map({True: 'none', False: 'specific'})
        stratify_column = df['_stratify_label']
        logger.info(f"Using stratified split for {domain} domain")
        logger.info(f"  'None of above': {(df['_stratify_label'] == 'none').sum()} samples")
        logger.info(f"  Specific answers: {(df['_stratify_label'] == 'specific').sum()} samples")
    
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_column
    )
    
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Remove temporary stratification column if it exists
    if '_stratify_label' in train_df.columns:
        train_df = train_df.drop(columns=['_stratify_label'])
        test_df = test_df.drop(columns=['_stratify_label'])
    
    # Combine
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Select important columns
    output_columns = [
        'prompt_id',
        'domain',
        'domain_key',
        'prompt',
        'formatted_prompt',
        'ground_truth',
        'split'
    ]
    
    # Keep only columns that exist
    output_columns = [col for col in output_columns if col in full_df.columns]
    output_df = full_df[output_columns]
    
    # Save processed data
    ensure_dir(output_dir)
    output_path = Path(output_dir) / f"{domain}_processed.csv"
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Saved processed dataset to {output_path}")
    logger.info(f"  Total: {len(output_df)} examples")
    logger.info(f"  Train: {len(train_df)} examples ({train_ratio*100:.0f}%)")
    logger.info(f"  Test: {len(test_df)} examples ({(1-train_ratio)*100:.0f}%)")
    
    return output_df


def process_all_datasets(output_dir: str = "data/processed"):
    """
    Process all available datasets.
    
    Args:
        output_dir: Output directory for processed data
    """
    domains = ['medicine', 'math', 'finance', 'is_agents', 'psychology']
    
    processed = []
    failed = []
    
    for domain in domains:
        try:
            df = process_dataset(domain, output_dir)
            processed.append((domain, len(df)))
        except FileNotFoundError as e:
            logger.warning(f"Skipping {domain}: {e}")
            failed.append(domain)
        except Exception as e:
            logger.error(f"Error processing {domain}: {e}")
            failed.append(domain)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    
    if processed:
        logger.info(f"\n✓ Successfully processed {len(processed)} datasets:")
        for domain, count in processed:
            logger.info(f"  - {domain}: {count} examples")
    
    if failed:
        logger.warning(f"\n✗ Failed to process {len(failed)} datasets:")
        for domain in failed:
            logger.warning(f"  - {domain}")
    
    # Create combined dataset
    if processed:
        logger.info("\nCreating combined dataset...")
        all_dfs = []
        
        for domain, _ in processed:
            path = Path(output_dir) / f"{domain}_processed.csv"
            df = pd.read_csv(path)
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = Path(output_dir) / "all_processed.csv"
        combined_df.to_csv(combined_path, index=False)
        
        logger.info(f"✓ Saved combined dataset to {combined_path}")
        logger.info(f"  Total: {len(combined_df)} examples across {len(processed)} domains")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process HHPF datasets")
    parser.add_argument(
        '--domain',
        type=str,
        choices=['medicine', 'math', 'finance', 'is_agents', 'psychology', 'all'],
        default='all',
        help='Domain to process (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    if args.domain == 'all':
        process_all_datasets(args.output_dir)
    else:
        process_dataset(args.domain, args.output_dir)


if __name__ == "__main__":
    main()
