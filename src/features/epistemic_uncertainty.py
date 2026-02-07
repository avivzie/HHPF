"""
Epistemic uncertainty features for HHPF.

Implements:
1. Semantic Entropy: Clustering of stochastic samples via NLI
2. Semantic Energy: Analysis of logit distribution
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import load_config, load_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticEntropyCalculator:
    """Calculate semantic entropy from stochastic samples using NLI."""
    
    def __init__(
        self,
        nli_model: str = "microsoft/deberta-base-mnli",
        device: Optional[str] = None
    ):
        """
        Initialize semantic entropy calculator.
        
        Args:
            nli_model: NLI model for entailment detection
            device: Device ('cuda', 'mps', 'cpu') or None for auto
        """
        self.config = load_config("features")['epistemic_uncertainty']['semantic_entropy']
        
        # Set device with MPS fallback for M1 Macs
        if device is None:
            try:
                if torch.backends.mps.is_available():
                    device = "mps"
                    # Test MPS device
                    test_tensor = torch.tensor([1.0]).to("mps")
                    logger.info("MPS device available and working")
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception as e:
                logger.warning(f"MPS failed, falling back to CPU: {e}")
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load NLI model (use MNLI-trained model for proper 3-class NLI)
        self.nli_model_name = nli_model or self.config.get('nli_model', 'microsoft/deberta-base-mnli')
        logger.info(f"Loading NLI model: {self.nli_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name
        ).to(self.device)
        self.model.eval()
        
        # Thresholds
        self.entailment_threshold = self.config.get('entailment_threshold', 0.7)
        self.batch_size = self.config.get('batch_size', 8)
    
    def compute_nli_score(self, text1: str, text2: str) -> float:
        """
        Compute NLI entailment score between two texts.
        
        Args:
            text1: First text (premise)
            text2: Second text (hypothesis)
            
        Returns:
            Entailment probability (0-1)
        """
        inputs = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # DeBERTa NLI: [contradiction, neutral, entailment]
        entailment_prob = probs[0, 2].item()
        
        return entailment_prob
    
    def compute_bidirectional_entailment(self, text1: str, text2: str) -> float:
        """
        Compute bidirectional entailment (semantic equivalence).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Average of both directions
        """
        score_12 = self.compute_nli_score(text1, text2)
        score_21 = self.compute_nli_score(text2, text1)
        
        return (score_12 + score_21) / 2.0
    
    def compute_nli_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise NLI scores for all texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            NxN matrix of bidirectional entailment scores
        """
        n = len(texts)
        matrix = np.zeros((n, n))
        
        # Diagonal is 1 (self-entailment)
        np.fill_diagonal(matrix, 1.0)
        
        # Compute upper triangle
        for i in range(n):
            for j in range(i + 1, n):
                score = self.compute_bidirectional_entailment(texts[i], texts[j])
                matrix[i, j] = score
                matrix[j, i] = score
        
        return matrix
    
    def cluster_by_entailment(
        self,
        texts: List[str],
        nli_matrix: Optional[np.ndarray] = None
    ) -> List[List[int]]:
        """
        Cluster texts by semantic equivalence.
        
        Args:
            texts: List of text samples
            nli_matrix: Precomputed NLI matrix (optional)
            
        Returns:
            List of clusters, each cluster is a list of indices
        """
        if len(texts) == 1:
            return [[0]]
        
        # Compute NLI matrix if not provided
        if nli_matrix is None:
            nli_matrix = self.compute_nli_matrix(texts)
        
        # Convert similarity to distance
        distance_matrix = 1.0 - nli_matrix
        
        # Use hierarchical clustering
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform clustering
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Cut tree at threshold
        distance_threshold = 1.0 - self.entailment_threshold
        cluster_labels = fcluster(
            linkage_matrix,
            distance_threshold,
            criterion='distance'
        )
        
        # Group indices by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        return list(clusters.values())
    
    def calculate_entropy(self, clusters: List[List[int]], total_samples: int) -> float:
        """
        Calculate entropy over cluster distribution.
        
        Args:
            clusters: List of clusters
            total_samples: Total number of samples
            
        Returns:
            Entropy value
        """
        if len(clusters) == 1:
            return 0.0  # No uncertainty
        
        # Calculate cluster probabilities
        probs = np.array([len(cluster) / total_samples for cluster in clusters])
        
        # Calculate entropy: H = -Σ p_i * log(p_i)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    def calculate_semantic_entropy(
        self,
        samples: List[str],
        return_clusters: bool = False
    ) -> Dict[str, any]:
        """
        Calculate semantic entropy from stochastic samples.
        
        Args:
            samples: List of generated text samples
            return_clusters: Whether to return cluster information
            
        Returns:
            Dictionary with entropy and metadata
        """
        if len(samples) == 0:
            raise ValueError("No samples provided")
        
        if len(samples) == 1:
            return {
                'semantic_entropy': 0.0,
                'num_samples': 1,
                'num_clusters': 1,
                'clusters': [[0]] if return_clusters else None
            }
        
        # Compute NLI matrix
        logger.debug(f"Computing NLI matrix for {len(samples)} samples")
        nli_matrix = self.compute_nli_matrix(samples)
        
        # Cluster samples
        clusters = self.cluster_by_entailment(samples, nli_matrix)
        
        # Calculate entropy
        entropy = self.calculate_entropy(clusters, len(samples))
        
        result = {
            'semantic_entropy': entropy,
            'num_samples': len(samples),
            'num_clusters': len(clusters),
            'avg_cluster_size': np.mean([len(c) for c in clusters]),
            'max_cluster_size': max(len(c) for c in clusters),
        }
        
        if return_clusters:
            result['clusters'] = clusters
            result['nli_matrix'] = nli_matrix
        
        return result


class NaiveConfidenceCalculator:
    """Calculate naive confidence baselines for comparison."""
    
    def __init__(self):
        """Initialize naive confidence calculator."""
        pass
    
    def calculate_max_prob(self, logprobs: Dict) -> float:
        """
        Calculate maximum probability (naive confidence baseline).
        
        Args:
            logprobs: Logprobs from model output
            
        Returns:
            Maximum token probability
        """
        token_logprobs = self._extract_token_logprobs(logprobs)
        
        if token_logprobs is None or len(token_logprobs) == 0:
            return None
        
        # Max probability = exp(max logprob)
        max_logprob = np.max(token_logprobs)
        max_prob = np.exp(max_logprob)
        
        return float(max_prob)
    
    def calculate_perplexity(self, logprobs: Dict) -> float:
        """
        Calculate token-level perplexity (naive confidence baseline).
        
        Perplexity = exp(-mean(log_probs))
        Lower perplexity = higher confidence
        
        Args:
            logprobs: Logprobs from model output
            
        Returns:
            Perplexity value
        """
        token_logprobs = self._extract_token_logprobs(logprobs)
        
        if token_logprobs is None or len(token_logprobs) == 0:
            return None
        
        # Perplexity = exp(-mean(logprobs))
        mean_logprob = np.mean(token_logprobs)
        perplexity = np.exp(-mean_logprob)
        
        return float(perplexity)
    
    def calculate_top_k_entropy(self, logprobs: Dict, k: int = 10) -> float:
        """
        Calculate entropy over top-k tokens (naive uncertainty).
        
        Args:
            logprobs: Logprobs from model output
            k: Number of top tokens to consider
            
        Returns:
            Entropy over top-k distribution
        """
        token_logprobs = self._extract_token_logprobs(logprobs)
        
        if token_logprobs is None or len(token_logprobs) == 0:
            return None
        
        # Average entropy across all positions
        entropies = []
        for logprob in token_logprobs:
            # For single token, entropy is 0
            # This is a simplified version - ideally we'd have top-k per position
            entropies.append(0.0)  # Placeholder
        
        return float(np.mean(entropies))
    
    def _extract_token_logprobs(self, logprobs: Dict) -> np.ndarray:
        """Extract token logprobs from API response."""
        if logprobs is None or (isinstance(logprobs, (list, np.ndarray)) and len(logprobs) == 0):
            return np.array([])
        
        # Handle dict format (OpenAI, Groq)
        if isinstance(logprobs, dict):
            if 'token_logprobs' in logprobs:
                token_logprobs = logprobs['token_logprobs']
            elif 'top_logprobs' in logprobs:
                token_logprobs = [
                    max(token_dict.values()) 
                    for token_dict in logprobs['top_logprobs']
                ]
            else:
                return np.array([])
        # Handle Together AI LogprobsPart object
        elif hasattr(logprobs, 'token_logprobs'):
            token_logprobs = logprobs.token_logprobs
        else:
            # Fallback: assume logprobs is already a list/array
            token_logprobs = logprobs
        
        # Convert to numpy and filter NaN
        logprobs_array = np.array(token_logprobs, dtype=float)
        logprobs_array = logprobs_array[~np.isnan(logprobs_array)]
        
        return logprobs_array
    
    def calculate_all_naive_metrics(self, logprobs: Dict) -> Dict[str, float]:
        """
        Calculate all naive confidence baselines.
        
        Args:
            logprobs: Logprobs from model output
            
        Returns:
            Dictionary with all naive metrics
        """
        return {
            'max_prob': self.calculate_max_prob(logprobs),
            'perplexity': self.calculate_perplexity(logprobs),
            'top_k_entropy': self.calculate_top_k_entropy(logprobs),
        }


class SemanticEnergyCalculator:
    """Calculate semantic energy from logit distribution."""
    
    def __init__(self):
        """Initialize semantic energy calculator."""
        self.config = load_config("features")['epistemic_uncertainty']['semantic_energy']
    
    def calculate_semantic_energy(
        self,
        logprobs: Dict,
        method: str = "negative_log_sum_exp"
    ) -> Dict[str, float]:
        """
        Calculate semantic energy from logit distribution.
        
        Args:
            logprobs: Logprobs from model output
            method: Calculation method
            
        Returns:
            Dictionary with energy metrics
        """
        if logprobs is None or (isinstance(logprobs, (list, np.ndarray)) and len(logprobs) == 0):
            logger.warning("No logprobs provided")
            return {
                'semantic_energy': None,
                'method': method
            }
        
        # Extract logits
        # Format depends on API provider
        # Together AI format: logprobs can be dict or LogprobsPart object with 'token_logprobs' attribute
        
        if isinstance(logprobs, dict):
            if 'token_logprobs' in logprobs:
                token_logprobs = logprobs['token_logprobs']
            elif 'top_logprobs' in logprobs:
                # Alternative format
                token_logprobs = [
                    max(token_dict.values()) 
                    for token_dict in logprobs['top_logprobs']
                ]
            else:
                logger.warning(f"Unknown logprobs format: {list(logprobs.keys())}")
                return {'semantic_energy': None, 'method': method}
        elif hasattr(logprobs, 'token_logprobs'):
            # Handle Together AI LogprobsPart object
            token_logprobs = logprobs.token_logprobs
        else:
            # Fallback: assume logprobs is already a list/array
            token_logprobs = logprobs
        
        if token_logprobs is None or (isinstance(token_logprobs, (list, np.ndarray)) and len(token_logprobs) == 0):
            return {'semantic_energy': None, 'method': method}
        
        # Convert to numpy array
        logprobs_array = np.array(token_logprobs, dtype=float)
        
        # Filter out None/NaN values
        logprobs_array = logprobs_array[~np.isnan(logprobs_array)]
        
        if len(logprobs_array) == 0:
            return {'semantic_energy': None, 'method': method}
        
        # Calculate energy
        if method == "negative_log_sum_exp":
            # E = -log(Σ exp(logit_i))
            # Numerically stable: E = -log(Σ exp(logit_i)) = -logsumexp(logits)
            from scipy.special import logsumexp
            energy = -logsumexp(logprobs_array)
            
        elif method == "mean_logprob":
            # Simple average of log probabilities
            energy = -np.mean(logprobs_array)
            
        elif method == "min_logprob":
            # Minimum log probability (least confident token)
            energy = -np.min(logprobs_array)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'semantic_energy': float(energy),
            'mean_logprob': float(np.mean(logprobs_array)),
            'min_logprob': float(np.min(logprobs_array)),
            'max_logprob': float(np.max(logprobs_array)),
            'std_logprob': float(np.std(logprobs_array)),
            'num_tokens': len(logprobs_array),
            'method': method
        }


class MCQFeatureCalculator:
    """Calculate MCQ-specific features for multiple-choice question formats.
    
    For MCQ responses, standard text-level features (NLI semantic entropy, 
    averaged logprobs) are orthogonal to correctness. Instead, we extract 
    features that capture the model's confidence about its letter choice:
    
    1. Letter Consistency: How often stochastic samples agree on the same letter
    2. MCQ Letter Entropy: Shannon entropy over the letter distribution
    3. First-Token Logprob: The logprob of just the answer letter token
    """
    
    # Valid MCQ option letters
    VALID_LETTERS = {'A', 'B', 'C', 'D'}
    
    @staticmethod
    def extract_selected_option(response: str) -> Optional[str]:
        """
        Extract the option letter (A/B/C/D) selected by the LLM.
        
        Mirrors the logic in MedicalLabeler.extract_selected_option.
        
        Args:
            response: LLM response text
            
        Returns:
            Option letter (A/B/C/D) or None if cannot parse
        """
        import re
        response = response.strip()
        
        # Check first few characters for single letter answer
        if len(response) > 0 and response[0].upper() in MCQFeatureCalculator.VALID_LETTERS:
            return response[0].upper()
        
        # Check for patterns like "Answer: A" or "The answer is B"
        patterns = [
            r'\b([ABCD])\b',
            r'answer[:\s]+([ABCD])\b',
            r'option[:\s]+([ABCD])\b',
            r'choice[:\s]+([ABCD])\b',
            r'^([ABCD])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def calculate_letter_consistency(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Calculate letter consistency across stochastic samples.
        
        Measures how often the stochastic samples agree on the same MCQ letter.
        High consistency = model is confident in its choice.
        
        Args:
            samples: List of response sample dicts with 'text' key
            
        Returns:
            Dictionary with letter consistency features
        """
        letters = []
        for sample in samples:
            text = sample.get('text', '')
            letter = self.extract_selected_option(text)
            if letter is not None:
                letters.append(letter)
        
        if len(letters) == 0:
            return {
                'mcq_letter_consistency': 0.0,
                'mcq_num_unique_letters': 0,
                'mcq_parseable_ratio': 0.0,
                'mcq_mode_letter_count': 0,
            }
        
        from collections import Counter
        letter_counts = Counter(letters)
        mode_count = letter_counts.most_common(1)[0][1]
        
        consistency = mode_count / len(letters)
        num_unique = len(letter_counts)
        parseable_ratio = len(letters) / len(samples)
        
        return {
            'mcq_letter_consistency': float(consistency),
            'mcq_num_unique_letters': num_unique,
            'mcq_parseable_ratio': float(parseable_ratio),
            'mcq_mode_letter_count': mode_count,
        }
    
    def calculate_mcq_letter_entropy(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Calculate Shannon entropy over the letter distribution from stochastic samples.
        
        Instead of NLI-based semantic entropy (which is broken for MCQ because
        explanations always differ), compute entropy over {A, B, C, D} choices.
        
        Args:
            samples: List of response sample dicts with 'text' key
            
        Returns:
            Dictionary with MCQ entropy features
        """
        letters = []
        for sample in samples:
            text = sample.get('text', '')
            letter = self.extract_selected_option(text)
            if letter is not None:
                letters.append(letter)
        
        if len(letters) == 0:
            return {
                'mcq_letter_entropy': np.log2(4),  # Maximum entropy as fallback
            }
        
        from collections import Counter
        letter_counts = Counter(letters)
        total = len(letters)
        
        # Shannon entropy over letter distribution
        probs = np.array([count / total for count in letter_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return {
            'mcq_letter_entropy': float(entropy),
        }
    
    def extract_first_token_logprob(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Extract the logprob of just the first content token (the answer letter).
        
        The first token carries the actual decision signal. Averaging over all
        ~1024 tokens drowns this signal in explanation text noise.
        
        Args:
            samples: List of response sample dicts (uses first sample's logprobs)
            
        Returns:
            Dictionary with first-token logprob features
        """
        primary_sample = samples[0] if samples else {}
        logprobs = primary_sample.get('logprobs')
        
        if logprobs is None:
            return {
                'mcq_first_token_logprob': None,
                'mcq_first_token_prob': None,
            }
        
        # Extract token logprobs array
        token_logprobs = None
        if isinstance(logprobs, dict):
            token_logprobs = logprobs.get('token_logprobs')
        elif hasattr(logprobs, 'token_logprobs'):
            token_logprobs = logprobs.token_logprobs
        elif isinstance(logprobs, (list, np.ndarray)):
            token_logprobs = logprobs
        
        if token_logprobs is None or len(token_logprobs) == 0:
            return {
                'mcq_first_token_logprob': None,
                'mcq_first_token_prob': None,
            }
        
        first_logprob = float(token_logprobs[0])
        
        # Handle NaN
        if np.isnan(first_logprob):
            return {
                'mcq_first_token_logprob': None,
                'mcq_first_token_prob': None,
            }
        
        return {
            'mcq_first_token_logprob': first_logprob,
            'mcq_first_token_prob': float(np.exp(first_logprob)),
        }
    
    def calculate_all_mcq_features(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Calculate all MCQ-specific features.
        
        Args:
            samples: List of response sample dicts
            
        Returns:
            Dictionary with all MCQ features
        """
        features = {}
        features.update(self.calculate_letter_consistency(samples))
        features.update(self.calculate_mcq_letter_entropy(samples))
        features.update(self.extract_first_token_logprob(samples))
        return features


def extract_epistemic_features(
    response_file: str,
    entropy_calc=None,
    energy_calc=None,
    calculate_entropy: bool = True,
    calculate_energy: bool = True,
    calculate_naive: bool = True,
    format_type: str = "free_text"
) -> Dict[str, any]:
    """
    Extract epistemic uncertainty features from response file.
    
    Args:
        response_file: Path to pickled response samples
        entropy_calc: Shared SemanticEntropyCalculator instance (optional)
        energy_calc: Shared SemanticEnergyCalculator instance (optional)
        calculate_entropy: Whether to calculate semantic entropy
        calculate_energy: Whether to calculate semantic energy
        calculate_naive: Whether to calculate naive confidence baselines
        format_type: Response format type ('free_text' or 'mcq')
        
    Returns:
        Dictionary with features
    """
    # Load samples
    samples = load_pickle(response_file)
    
    features = {}
    
    # MCQ-specific features (for multiple-choice format)
    if format_type == "mcq":
        mcq_calc = MCQFeatureCalculator()
        mcq_features = mcq_calc.calculate_all_mcq_features(samples)
        features.update(mcq_features)
    
    # Semantic Entropy (standard NLI-based)
    if calculate_entropy:
        texts = [sample['text'] for sample in samples]
        
        # Use shared instance if provided, otherwise create new one
        if entropy_calc is None:
            entropy_calc = SemanticEntropyCalculator()
        
        entropy_result = entropy_calc.calculate_semantic_entropy(texts)
        
        features.update({
            'semantic_entropy': entropy_result['semantic_entropy'],
            'num_semantic_clusters': entropy_result['num_clusters'],
            'avg_cluster_size': entropy_result['avg_cluster_size'],
        })
    
    # Semantic Energy
    if calculate_energy:
        # Use first sample (primary response) for logprobs
        primary_sample = samples[0]
        
        if 'logprobs' in primary_sample:
            energy_calc = SemanticEnergyCalculator()
            energy_result = energy_calc.calculate_semantic_energy(
                primary_sample['logprobs']
            )
            
            features.update({
                'semantic_energy': energy_result['semantic_energy'],
                'mean_logprob': energy_result.get('mean_logprob'),
                'min_logprob': energy_result.get('min_logprob'),
                'std_logprob': energy_result.get('std_logprob'),
            })
        else:
            logger.warning("No logprobs found in primary sample")
            features.update({
                'semantic_energy': None,
                'mean_logprob': None,
                'min_logprob': None,
                'std_logprob': None,
            })
    
    # Naive Confidence Baselines (for RQ2 comparison)
    if calculate_naive:
        primary_sample = samples[0]
        
        if 'logprobs' in primary_sample:
            naive_calc = NaiveConfidenceCalculator()
            naive_metrics = naive_calc.calculate_all_naive_metrics(
                primary_sample['logprobs']
            )
            
            features.update({
                'naive_max_prob': naive_metrics['max_prob'],
                'naive_perplexity': naive_metrics['perplexity'],
                'naive_top_k_entropy': naive_metrics['top_k_entropy'],
            })
        else:
            features.update({
                'naive_max_prob': None,
                'naive_perplexity': None,
                'naive_top_k_entropy': None,
            })
    
    return features
