"""
Label responses with hallucination labels BEFORE train/test split.

This module handles the critical step of labeling all responses before
splitting data, ensuring proper stratification is possible.
"""

import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import List, Dict
from sklearn.model_selection import train_test_split

from src.data_preparation.ground_truth import get_labeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def label_all_responses(
    processed_csv: str,
    responses_dir: str,
    domain: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Label all responses and create stratified train/test split.
    
    This function:
    1. Loads all response files
    2. Labels each response with hallucination label
    3. Creates stratified train/test split based on labels
    4. Returns labeled DataFrame with split column
    
    Args:
        processed_csv: Path to processed CSV with prompts/ground truth
        responses_dir: Directory containing response pickle files
        domain: Domain name for labeling
        train_ratio: Train/test split ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with labeled responses and train/test split
    """
    logger.info(f"Labeling all responses for {domain} domain...")
    
    # Load processed data
    df = pd.read_csv(processed_csv)
    logger.info(f"Loaded {len(df)} samples from {processed_csv}")
    
    # Get labeler for domain
    labeler = get_labeler(domain)
    
    # Load and label all responses
    responses_dir = Path(responses_dir)
    all_data = []
    
    for idx, row in df.iterrows():
        prompt_id = row['prompt_id']
        response_file = responses_dir / f"{prompt_id}_responses.pkl"
        
        if not response_file.exists():
            logger.warning(f"Missing response file for {prompt_id}, skipping")
            continue
        
        try:
            # Load responses
            with open(response_file, 'rb') as f:
                responses = pickle.load(f)
            
            # Use first response as primary
            if isinstance(responses, list) and len(responses) > 0:
                if isinstance(responses[0], dict):
                    primary_response = responses[0].get('text', responses[0])
                else:
                    primary_response = responses[0]
            else:
                primary_response = str(responses)
            
            # Label the response
            label_result = labeler.label_response(
                response=primary_response,
                ground_truth=row['ground_truth'],
                domain=domain,
                prompt=row['prompt']
            )
            
            # Create record
            record = {
                'prompt_id': prompt_id,
                'domain': row['domain'],
                'prompt': row['prompt'],
                'ground_truth': row['ground_truth'],
                'primary_response': primary_response,
                'hallucination_label': label_result['hallucination_label'],
                'label_confidence': label_result.get('confidence', 0.0),
                'label_method': label_result.get('method', 'unknown'),
                'response_file': str(response_file)
            }
            
            # Add domain-specific label fields
            for key, value in label_result.items():
                if key not in ['hallucination_label', 'confidence', 'method']:
                    record[key] = value
            
            all_data.append(record)
            
        except Exception as e:
            logger.warning(f"Failed to process {prompt_id}: {e}")
            continue
    
    logger.info(f"✓ Labeled {len(all_data)} samples")
    
    # Convert to DataFrame
    labeled_df = pd.DataFrame(all_data)
    
    # Check label distribution
    hall_count = (labeled_df['hallucination_label'] == 1).sum()
    faith_count = (labeled_df['hallucination_label'] == 0).sum()
    logger.info(f"\nLabel distribution:")
    logger.info(f"  Hallucinations: {hall_count} ({hall_count/len(labeled_df)*100:.1f}%)")
    logger.info(f"  Faithful: {faith_count} ({faith_count/len(labeled_df)*100:.1f}%)")
    
    # Stratified train/test split on actual labels
    logger.info(f"\nCreating stratified split (train={train_ratio:.0%}, test={1-train_ratio:.0%})...")
    
    if len(labeled_df['hallucination_label'].unique()) < 2:
        logger.error("Cannot stratify - only one class present!")
        raise ValueError("Need both classes for stratified split")
    
    train_df, test_df = train_test_split(
        labeled_df,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=labeled_df['hallucination_label']  # ✅ STRATIFY ON ACTUAL LABELS!
    )
    
    # Add split column
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Combine
    final_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Verify stratification worked
    train_hall = (train_df['hallucination_label'] == 1).sum() / len(train_df) * 100
    test_hall = (test_df['hallucination_label'] == 1).sum() / len(test_df) * 100
    gap = abs(train_hall - test_hall)
    
    logger.info(f"\n✓ Stratified split complete:")
    logger.info(f"  Train: {len(train_df)} samples, {train_hall:.1f}% hallucinations")
    logger.info(f"  Test:  {len(test_df)} samples, {test_hall:.1f}% hallucinations")
    logger.info(f"  Train/Test gap: {gap:.1f} percentage points")
    
    if gap > 5.0:
        logger.warning(f"⚠️  Train/test gap is {gap:.1f}% (should be <5%)")
    else:
        logger.info(f"  ✓ Gap is acceptable (<5%)")
    
    return final_df


def save_labeled_responses(labeled_df: pd.DataFrame, output_path: str):
    """
    Save labeled responses to CSV.
    
    Args:
        labeled_df: DataFrame with labeled responses
        output_path: Path to save CSV
    """
    labeled_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved labeled responses to {output_path}")
