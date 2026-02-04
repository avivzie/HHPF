"""
Response generation with stochastic sampling for semantic entropy calculation.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from tqdm import tqdm

from src.inference.llama_client import LlamaClient
from src.utils import load_config, ensure_dir, save_pickle, load_pickle
from src.data_preparation.ground_truth import get_labeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses with stochastic sampling."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "together",
        enable_cache: bool = True
    ):
        """
        Initialize response generator.
        
        Args:
            model: Llama model name
            provider: API provider
            enable_cache: Enable caching
        """
        self.client = LlamaClient(model, provider, enable_cache)
        self.config = load_config("model")['llama']
    
    def generate_single_response(
        self,
        prompt: str,
        temperature: float = 0.8,
        extract_logprobs: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a single response with logprobs.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            extract_logprobs: Whether to extract logprobs
            
        Returns:
            Response dictionary
        """
        logprobs = self.config.get('logprobs', 100) if extract_logprobs else None
        
        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            logprobs=logprobs
        )
        
        return response
    
    def generate_stochastic_samples(
        self,
        prompt: str,
        num_samples: Optional[int] = None,
        temperature_range: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple stochastic samples for semantic entropy.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples (default from config)
            temperature_range: Temperature range for sampling
            
        Returns:
            List of response dictionaries
        """
        num_samples = num_samples or self.config.get('num_samples', 10)
        temp_range = temperature_range or self.config.get('sample_temperature_range', [0.7, 1.0])
        
        # Generate temperatures
        if num_samples == 1:
            temperatures = [np.mean(temp_range)]
        else:
            temperatures = np.linspace(temp_range[0], temp_range[1], num_samples)
        
        samples = []
        
        for i, temp in enumerate(temperatures):
            logger.debug(f"Generating sample {i+1}/{num_samples} (T={temp:.2f})")
            
            response = self.generate_single_response(
                prompt,
                temperature=temp,
                extract_logprobs=(i == 0)  # Only extract logprobs for first sample
            )
            
            response['sample_id'] = i
            response['temperature'] = temp
            samples.append(response)
        
        return samples
    
    def generate_for_dataset(
        self,
        dataset_path: str,
        output_dir: str = "data/features",
        num_samples: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate responses for an entire dataset.
        
        Args:
            dataset_path: Path to processed dataset CSV
            output_dir: Output directory for responses
            num_samples: Number of stochastic samples per prompt
            limit: Limit number of prompts (for testing)
            
        Returns:
            DataFrame with generated responses
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} prompts from {dataset_path}")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} prompts for testing")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Create progress log file
        progress_log_path = Path(output_dir) / "progress_log.txt"
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        def log_progress(message: str):
            """Log to both console and progress file."""
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {message}"
            logger.info(log_msg)
            with open(progress_log_path, "a") as f:
                f.write(log_msg + "\n")
                f.flush()  # Force write to disk
        
        log_progress(f"Starting response generation for {len(df)} prompts")
        
        # Generate responses
        results = []
        start_time = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
            prompt_id = row['prompt_id']
            prompt = row['formatted_prompt']
            ground_truth = row['ground_truth']
            domain = row['domain_key']
            
            # Log progress
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(df) - idx - 1) / rate if rate > 0 else 0
            
            log_progress(
                f"Processing {prompt_id} ({idx+1}/{len(df)}) | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"Rate: {rate*60:.1f}/min | "
                f"ETA: {remaining/60:.1f}m"
            )
            
            # Check if already generated
            cache_path = Path(output_dir) / f"{prompt_id}_responses.pkl"
            
            if cache_path.exists():
                logger.debug(f"Loading cached responses for {prompt_id}")
                samples = load_pickle(cache_path)
            else:
                # Generate stochastic samples
                samples = self.generate_stochastic_samples(
                    prompt,
                    num_samples=num_samples
                )
                
                # Save samples
                save_pickle(samples, cache_path)
            
            # Extract primary response (first sample)
            primary_response = samples[0]['text']
            
            # Label response
            labeler = get_labeler(domain)
            
            # Check if dataset has existing labels (e.g., HalluMix)
            existing_label = row.get('existing_label', None)
            
            if existing_label is not None:
                # Use existing label for datasets that have them
                label_result = labeler.label_response(
                    primary_response,
                    ground_truth,
                    domain,
                    existing_label=existing_label
                )
            else:
                # Generate label by comparing response to ground truth
                label_result = labeler.label_response(
                    primary_response,
                    ground_truth,
                    domain
                )
            
            # Collect result
            result = {
                'prompt_id': prompt_id,
                'domain': domain,
                'prompt': prompt,
                'ground_truth': ground_truth,
                'split': row['split'],  # Preserve train/test split
                'primary_response': primary_response,
                'num_samples': len(samples),
                'hallucination_label': label_result['hallucination_label'],
                'label_confidence': label_result.get('confidence', 1.0),
                'label_method': label_result.get('method', 'unknown'),
                'response_file': str(cache_path)
            }
            
            # Add labeling metadata
            for key, value in label_result.items():
                if key not in ['hallucination_label', 'confidence', 'method']:
                    result[f'label_{key}'] = value
            
            results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = Path(output_dir) / f"responses_{Path(dataset_path).stem}.csv"
        results_df.to_csv(output_path, index=False)
        
        # Log completion
        total_time = time.time() - start_time
        log_progress(f"✓ Generation complete! Total time: {total_time/60:.1f}m")
        log_progress(f"  Saved responses to {output_path}")
        log_progress(f"  Total responses: {len(results_df)}")
        log_progress(f"  Hallucinations: {results_df['hallucination_label'].sum()} "
                    f"({results_df['hallucination_label'].mean()*100:.1f}%)")
        
        logger.info(f"✓ Saved responses to {output_path}")
        logger.info(f"  Total responses: {len(results_df)}")
        logger.info(f"  Hallucinations: {results_df['hallucination_label'].sum()} "
                   f"({results_df['hallucination_label'].mean()*100:.1f}%)")
        
        return results_df


def main():
    """Main entry point for response generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Llama-3 responses")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name or path to processed CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['llama-3-8b', 'llama-3-70b'],
        default='llama-3-8b',
        help='Model size'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['together', 'groq'],
        default='together',
        help='API provider'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of stochastic samples per prompt'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of prompts (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset.endswith('.csv'):
        dataset_path = args.dataset
    else:
        dataset_path = f"data/processed/{args.dataset}_processed.csv"
    
    # Map model names
    model_map = {
        'llama-3-8b': 'meta-llama/Llama-3-8b-chat-hf',
        'llama-3-70b': 'meta-llama/Llama-3-70b-chat-hf',
    }
    model = model_map.get(args.model, args.model)
    
    # Generate responses
    generator = ResponseGenerator(model=model, provider=args.provider)
    
    generator.generate_for_dataset(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        limit=args.limit
    )
    
    logger.info("✓ Response generation complete!")


if __name__ == "__main__":
    main()
