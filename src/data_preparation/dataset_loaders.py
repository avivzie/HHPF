"""
Dataset loaders for HHPF.

Handles loading and parsing of raw datasets from different sources.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.utils import load_config, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for loading datasets."""
    
    def __init__(self, config_name: str = "datasets"):
        """Initialize loader with configuration."""
        self.config = load_config(config_name)
        self.datasets_config = self.config['datasets']
        self.global_config = self.config['global']
        
    def load_dataset(self, domain: str) -> pd.DataFrame:
        """
        Load a specific dataset by domain.
        
        Args:
            domain: Domain name (medicine, math, finance, is_agents, psychology)
            
        Returns:
            DataFrame with raw dataset
        """
        if domain not in self.datasets_config:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.datasets_config.keys())}")
        
        dataset_config = self.datasets_config[domain]
        file_path = Path(dataset_config['file_path'])
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Please place the {dataset_config['name']} dataset in data/raw/"
            )
        
        logger.info(f"Loading {dataset_config['name']} from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Add metadata
        df['domain'] = dataset_config['domain']
        df['domain_key'] = domain
        
        logger.info(f"Loaded {len(df)} examples from {dataset_config['name']}")
        
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary mapping domain name to DataFrame
        """
        datasets = {}
        
        for domain in self.datasets_config.keys():
            try:
                datasets[domain] = self.load_dataset(domain)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {domain}: {e}")
                continue
        
        return datasets
    
    def get_prompt_and_answer(self, domain: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract prompt and answer columns based on domain configuration.
        
        Args:
            domain: Domain name
            df: Raw dataset DataFrame
            
        Returns:
            DataFrame with 'prompt' and 'ground_truth' columns
        """
        dataset_config = self.datasets_config[domain]
        
        prompt_col = dataset_config['prompt_column']
        answer_col = dataset_config['answer_column']
        
        # Filter out rows with missing prompts (important for HalluMix)
        if prompt_col in df.columns:
            df = df[df[prompt_col].notna()].copy()
            logger.info(f"Filtered to {len(df)} samples with valid prompts")
        
        if prompt_col not in df.columns:
            raise ValueError(f"Prompt column '{prompt_col}' not found in {domain} dataset")
        
        if answer_col not in df.columns:
            raise ValueError(f"Answer column '{answer_col}' not found in {domain} dataset")
        
        # Create standardized columns
        result = df.copy()
        result['prompt'] = result[prompt_col]
        result['ground_truth'] = result[answer_col]
        
        # If dataset has pre-existing hallucination labels, preserve them
        if dataset_config.get('has_labels', False) and 'hallucination_label' in result.columns:
            # Convert boolean to int (True->1, False->0)
            result['existing_label'] = result['hallucination_label'].astype(int)
            logger.info(f"Found existing hallucination labels for {domain}")
        
        # Create unique prompt IDs
        result['prompt_id'] = result.apply(
            lambda row: f"{domain}_{row.name}", axis=1
        )
        
        return result
    
    def validate_dataset(self, df: pd.DataFrame, domain: str) -> bool:
        """
        Validate dataset has required columns and format.
        
        Args:
            df: Dataset DataFrame
            domain: Domain name
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_cols = ['prompt', 'ground_truth', 'prompt_id', 'domain']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns in {domain}: {missing}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found in {domain}:\n{null_counts[null_counts > 0]}")
        
        # Check minimum samples
        min_samples = self.global_config.get('min_samples_per_domain', 500)
        if len(df) < min_samples:
            logger.warning(
                f"{domain} has only {len(df)} samples (minimum recommended: {min_samples})"
            )
        
        logger.info(f"✓ {domain} dataset validated: {len(df)} samples")
        
        return True


class MedicineLoader(DatasetLoader):
    """Specialized loader for Med-HALT dataset."""
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Med-HALT dataset."""
        # Filter out samples with NULL ground truth
        # Med-HALT has ~1,860 samples with missing ground truth
        initial_count = len(df)
        df = df[df['ground_truth'].notna() & (df['ground_truth'] != '')]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} samples with NULL ground truth ({filtered_count/initial_count*100:.1f}%)")
        
        return df


class MathLoader(DatasetLoader):
    """Specialized loader for GSM8K dataset."""
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess GSM8K dataset."""
        # Extract numerical answer if needed
        # GSM8K answers often in format "#### 42"
        def extract_numerical_answer(answer_text):
            if pd.isna(answer_text):
                return None
            
            answer_str = str(answer_text)
            
            # Check for #### format
            if '####' in answer_str:
                parts = answer_str.split('####')
                if len(parts) > 1:
                    return parts[-1].strip()
            
            return answer_str
        
        df['ground_truth'] = df['ground_truth'].apply(extract_numerical_answer)
        
        return df


class FinanceLoader(DatasetLoader):
    """Specialized loader for FinanceBench dataset."""
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess FinanceBench dataset."""
        # Add any FinanceBench specific preprocessing
        return df


class ISAgentsLoader(DatasetLoader):
    """Specialized loader for HalluMix dataset."""
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess HalluMix dataset."""
        # Add any HalluMix specific preprocessing
        return df


class PsychologyLoader(DatasetLoader):
    """Specialized loader for TruthfulQA dataset."""
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess TruthfulQA dataset."""
        # TruthfulQA may have multiple correct/incorrect answers
        # Use best_answer as ground truth
        return df


def get_loader(domain: str) -> DatasetLoader:
    """
    Get appropriate loader for domain.
    
    Args:
        domain: Domain name
        
    Returns:
        Specialized DatasetLoader instance
    """
    loaders = {
        'medicine': MedicineLoader,
        'math': MathLoader,
        'finance': FinanceLoader,
        'is_agents': ISAgentsLoader,
        'psychology': PsychologyLoader,
    }
    
    loader_class = loaders.get(domain, DatasetLoader)
    return loader_class()
