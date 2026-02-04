"""
Feature aggregator for HHPF.

Combines all features (epistemic + contextual + domain) into unified training matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
from tqdm import tqdm

from src.features.epistemic_uncertainty import extract_epistemic_features
from src.features.contextual_features import extract_contextual_features
from src.utils import load_config, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregate all features for model training."""
    
    def __init__(self):
        """Initialize feature aggregator."""
        self.config = load_config("features")['aggregation']
        self.domain_config = load_config("features")['domain']
        
        # Initialize shared calculators (lazy load on first use)
        self.entropy_calc = None
        self.energy_calc = None
    
    def extract_features_for_prompt(
        self,
        prompt_id: str,
        prompt: str,
        response_file: str,
        domain: str
    ) -> Dict[str, any]:
        """
        Extract all features for a single prompt.
        
        Args:
            prompt_id: Prompt identifier
            prompt: Prompt text
            response_file: Path to response samples file
            domain: Domain name
            
        Returns:
            Dictionary with all features
        """
        features = {
            'prompt_id': prompt_id,
            'domain': domain
        }
        
        # Epistemic uncertainty features
        try:
            # Lazy initialize calculators on first use
            if self.entropy_calc is None:
                logger.info("Initializing SemanticEntropyCalculator (one-time)")
                from src.features.epistemic_uncertainty import SemanticEntropyCalculator
                self.entropy_calc = SemanticEntropyCalculator()
            
            if self.energy_calc is None:
                logger.info("Initializing SemanticEnergyCalculator (one-time)")
                from src.features.epistemic_uncertainty import SemanticEnergyCalculator
                self.energy_calc = SemanticEnergyCalculator()
            
            # Pass shared instances to avoid reloading models
            epistemic_features = extract_epistemic_features(
                response_file,
                entropy_calc=self.entropy_calc,
                energy_calc=self.energy_calc
            )
            features.update(epistemic_features)
        except Exception as e:
            logger.warning(f"Failed to extract epistemic features for {prompt_id}: {e}")
            features.update({
                'semantic_entropy': None,
                'semantic_energy': None,
                'num_semantic_clusters': None,
            })
        
        # Contextual features
        try:
            contextual_features = extract_contextual_features(prompt)
            features.update(contextual_features)
        except Exception as e:
            logger.warning(f"Failed to extract contextual features for {prompt_id}: {e}")
            features.update({
                'avg_entity_rarity': None,
                'num_entities': 0,
                'token_count': 0,
                'lexical_diversity': None,
            })
        
        return features
    
    def encode_domain(self, domain: str) -> Dict[str, int]:
        """
        Encode domain as one-hot or label.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with encoded domain
        """
        encoding = self.domain_config.get('encoding', 'onehot')
        domains = self.domain_config.get('domains', [])
        
        if encoding == 'onehot':
            # One-hot encoding
            result = {f'domain_{d}': 0 for d in domains}
            
            # Find matching domain
            for d in domains:
                if d in domain or d.lower() in domain.lower():
                    result[f'domain_{d}'] = 1
                    break
            
            return result
        
        elif encoding == 'label':
            # Label encoding
            try:
                label = domains.index(domain)
            except ValueError:
                label = -1
            
            return {'domain_label': label}
        
        else:
            return {}
    
    def aggregate_features(
        self,
        responses_csv: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate features for all prompts in a dataset.
        
        Args:
            responses_csv: Path to responses CSV (from response_generator)
            output_path: Output path for feature matrix
            
        Returns:
            DataFrame with all features
        """
        # Load responses
        responses_df = pd.read_csv(responses_csv)
        logger.info(f"Loaded {len(responses_df)} responses from {responses_csv}")
        
        # Extract features for each prompt
        all_features = []
        
        for idx, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Extracting features"):
            features = self.extract_features_for_prompt(
                prompt_id=row['prompt_id'],
                prompt=row['prompt'],
                response_file=row['response_file'],
                domain=row['domain']
            )
            
            # Add domain encoding
            domain_features = self.encode_domain(row['domain'])
            features.update(domain_features)
            
            # Add label
            features['hallucination_label'] = row['hallucination_label']
            
            # Add metadata
            if self.config.get('include_metadata', True):
                features['ground_truth'] = row.get('ground_truth', '')
                features['primary_response'] = row.get('primary_response', '')
                features['split'] = row.get('split', 'train')
            
            all_features.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Save if output path provided
        if output_path:
            ensure_dir(Path(output_path).parent)
            
            output_format = self.config.get('output_format', 'csv')
            
            if output_format == 'csv':
                features_df.to_csv(output_path, index=False)
            elif output_format == 'parquet':
                features_df.to_parquet(output_path, index=False)
            elif output_format == 'feather':
                features_df.to_feather(output_path)
            
            logger.info(f"✓ Saved feature matrix to {output_path}")
            logger.info(f"  Shape: {features_df.shape}")
            logger.info(f"  Features: {features_df.shape[1] - 1}")  # Exclude label
        
        return features_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in feature matrix.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        method = self.config.get('handle_missing', 'mean')
        
        # Identify numeric columns (features)
        metadata_cols = ['prompt_id', 'domain', 'ground_truth', 'primary_response', 
                        'split', 'hallucination_label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif method == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata and label).
        
        Args:
            df: Feature DataFrame
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['prompt_id', 'domain', 'ground_truth', 'primary_response', 
                       'split', 'hallucination_label']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


def main():
    """Main entry point for feature aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate HHPF features")
    parser.add_argument(
        '--responses',
        type=str,
        required=True,
        help='Path to responses CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for feature matrix'
    )
    
    args = parser.parse_args()
    
    # Aggregate features
    aggregator = FeatureAggregator()
    features_df = aggregator.aggregate_features(
        responses_csv=args.responses,
        output_path=args.output
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(features_df)}")
    logger.info(f"Total features: {len(aggregator.get_feature_columns(features_df))}")
    logger.info(f"Hallucinations: {features_df['hallucination_label'].sum()} "
               f"({features_df['hallucination_label'].mean()*100:.1f}%)")
    
    # Feature statistics
    feature_cols = aggregator.get_feature_columns(features_df)
    logger.info(f"\nFeature statistics:")
    for col in feature_cols[:10]:  # Show first 10
        if col in features_df.columns:
            values = features_df[col].dropna()
            if len(values) > 0:
                logger.info(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}")


if __name__ == "__main__":
    main()
