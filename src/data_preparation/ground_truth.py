"""
Ground truth extraction and hallucination labeling for HHPF.

Determines whether a generated response is a hallucination based on domain-specific criteria.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from difflib import SequenceMatcher

from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthLabeler:
    """Base class for ground truth labeling."""
    
    def __init__(self, config_name: str = "datasets"):
        """Initialize labeler with configuration."""
        self.config = load_config(config_name)
        self.datasets_config = self.config['datasets']
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label a response as hallucinated or faithful.
        
        Args:
            response: Generated response text
            ground_truth: Ground truth answer
            domain: Domain name
            prompt: Optional prompt/question text (for context-aware labeling)
            **kwargs: Additional domain-specific arguments
            
        Returns:
            Dictionary with 'hallucination_label' (0/1) and 'confidence'
        """
        raise NotImplementedError("Subclasses must implement label_response")
    
    def get_labeler(self, domain: str) -> 'GroundTruthLabeler':
        """Get appropriate labeler for domain."""
        labelers = {
            'medicine': MedicalLabeler,
            'math': MathLabeler,
            'finance': FinanceLabeler,
            'is_agents': FactualConsistencyLabeler,
            'psychology': TruthfulnessLabeler,
        }
        
        labeler_class = labelers.get(domain, GroundTruthLabeler)
        return labeler_class()


class MathLabeler(GroundTruthLabeler):
    """Labeler for mathematical problems (GSM8K)."""
    
    def extract_numerical_value(self, text: str) -> Optional[float]:
        """
        Extract numerical value from text.
        
        Args:
            text: Text containing number
            
        Returns:
            Extracted float value or None
        """
        if pd.isna(text) or not text:
            return None
        
        text = str(text).strip()
        
        # Remove common text patterns
        text = text.replace('$', '').replace(',', '').replace('%', '')
        
        # Find all numbers (including decimals)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                # Return the last number found (usually the final answer)
                return float(numbers[-1])
            except ValueError:
                return None
        
        return None
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label math response based on numerical accuracy.
        
        Returns:
            Dict with 'hallucination_label' (1 if wrong, 0 if correct) and metadata
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in math domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        response_value = self.extract_numerical_value(response)
        truth_value = self.extract_numerical_value(str(ground_truth))
        
        if response_value is None or truth_value is None:
            logger.warning(f"Could not extract numerical values: response={response_value}, truth={truth_value}")
            return {
                'hallucination_label': 1,  # Conservative: label as hallucination
                'confidence': 0.5,
                'extracted_response': response_value,
                'extracted_truth': truth_value,
                'method': 'numerical_extraction'
            }
        
        # Check if values match (with small tolerance for floating point)
        tolerance = 1e-6
        matches = abs(response_value - truth_value) < tolerance
        
        return {
            'hallucination_label': 0 if matches else 1,
            'confidence': 1.0,
            'extracted_response': response_value,
            'extracted_truth': truth_value,
            'difference': abs(response_value - truth_value),
            'method': 'exact_numerical_match'
        }


class FinanceLabeler(GroundTruthLabeler):
    """Labeler for financial questions (FinanceBench)."""
    
    def extract_numerical_value_with_units(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract numerical value and detect unit (thousand, million, billion, trillion).
        
        Args:
            text: Text containing number with possible unit
            
        Returns:
            Tuple of (normalized_value, unit_found)
        """
        if pd.isna(text) or not text:
            return None, None
        
        text_lower = str(text).lower()
        
        # Unit multipliers
        units = {
            'trillion': 1e12,
            'billion': 1e9,
            'million': 1e6,
            'thousand': 1e3,
            'bn': 1e9,
            'mn': 1e6,
            'k': 1e3,
        }
        
        # Extract numerical value
        math_labeler = MathLabeler()
        base_value = math_labeler.extract_numerical_value(text)
        
        if base_value is None:
            return None, None
        
        # Check for unit indicators
        detected_unit = None
        multiplier = 1.0
        
        for unit, mult in units.items():
            if unit in text_lower:
                detected_unit = unit
                multiplier = mult
                break
        
        normalized_value = base_value * multiplier
        
        return normalized_value, detected_unit
    
    def infer_unit_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Infer the expected unit from the prompt/question.
        
        Args:
            prompt: The question text
            
        Returns:
            Unit string ('million', 'billion', etc.) or None
        """
        if not prompt:
            return None
        
        prompt_lower = prompt.lower()
        
        # Check for explicit unit specifications in order of priority
        unit_patterns = [
            ('usd millions', 'million'),
            ('usd million', 'million'),
            ('in millions', 'million'),
            ('usd billions', 'billion'),
            ('usd billion', 'billion'),
            ('in billions', 'billion'),
            ('in thousands', 'thousand'),
            ('usd thousands', 'thousand'),
        ]
        
        for pattern, unit in unit_patterns:
            if pattern in prompt_lower:
                return unit
        
        return None
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label finance response with unit-aware numerical tolerance and relaxed text similarity.
        
        Args:
            prompt: Optional question text to infer expected units
        
        Returns:
            Dict with 'hallucination_label' and metadata
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in finance domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        # Get tolerance from config
        dataset_config = self.datasets_config.get(domain, {})
        tolerance_pct = dataset_config.get('tolerance', 0.05)  # Increased to 5% for finance
        
        # Try unit-aware numerical comparison
        response_value, response_unit = self.extract_numerical_value_with_units(response)
        truth_value, truth_unit = self.extract_numerical_value_with_units(str(ground_truth))
        
        # If ground truth has no explicit unit but prompt specifies one, infer it
        if truth_value is not None and truth_unit is None and prompt:
            inferred_unit = self.infer_unit_from_prompt(prompt)
            if inferred_unit:
                # Apply the inferred unit to ground truth
                unit_multipliers = {
                    'trillion': 1e12,
                    'billion': 1e9,
                    'million': 1e6,
                    'thousand': 1e3,
                }
                truth_value = truth_value * unit_multipliers.get(inferred_unit, 1.0)
                truth_unit = f'{inferred_unit} (inferred)'
        
        # Check if this is truly a numerical answer or a qualitative one with embedded numbers
        # Heuristic: If ground truth is long (>50 chars) with multiple sentences, it's qualitative
        is_qualitative = len(str(ground_truth)) > 50 or '\n' in str(ground_truth) or any(word in str(ground_truth).lower() for word in ['yes', 'no', 'because', 'due to', 'primarily', 'evident from'])
        
        if response_value is not None and truth_value is not None and not is_qualitative:
            # Both values are now normalized to same base (no units)
            # Check with percentage tolerance
            if truth_value == 0:
                tolerance_abs = 1e-6
            else:
                tolerance_abs = abs(truth_value * tolerance_pct)
            
            difference = abs(response_value - truth_value)
            matches = difference <= tolerance_abs
            
            return {
                'hallucination_label': 0 if matches else 1,
                'confidence': 1.0,
                'extracted_response': response_value,
                'extracted_truth': truth_value,
                'response_unit': response_unit,
                'truth_unit': truth_unit,
                'tolerance_used': tolerance_abs,
                'difference': difference,
                'method': 'numerical_tolerance_unit_aware'
            }
        
        # Fallback to text similarity with relaxed threshold
        # Finance answers can be phrased differently but still be correct
        similarity = SequenceMatcher(None, response.lower(), ground_truth.lower()).ratio()
        
        # Also check for key term overlap for qualitative answers
        response_terms = set(response.lower().split())
        truth_terms = set(ground_truth.lower().split())
        
        # Remove common stopwords for better matching
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'to', 'of', 'and', 'in', 'for', 'on', 'with'}
        response_terms_filtered = response_terms - stopwords
        truth_terms_filtered = truth_terms - stopwords
        
        if truth_terms_filtered:
            term_overlap = len(response_terms_filtered & truth_terms_filtered) / len(truth_terms_filtered)
        else:
            term_overlap = 0.0
        
        # Combined score for qualitative answers
        combined_score = (similarity * 0.6) + (term_overlap * 0.4)
        
        # Very relaxed threshold for finance qualitative answers (0.35)
        # Finance questions can have many valid phrasings of the same conclusion
        threshold = 0.35
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.7,
            'similarity_score': similarity,
            'term_overlap': term_overlap,
            'combined_score': combined_score,
            'method': 'text_similarity_with_term_overlap'
        }


class MedicalLabeler(GroundTruthLabeler):
    """Labeler for medical questions (Med-HALT)."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label medical response using text similarity and key term matching.
        
        For research validity, this should ideally use medical NLI or expert validation.
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in medical domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,  # Conservative: mark as hallucination if no GT
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        response_lower = response.lower()
        truth_lower = str(ground_truth).lower()
        
        # Special handling for "None of the above" answers
        # These require exact semantic matching - if GT says "none", response must also say "none"
        # Otherwise it's a hallucination (LLM provided specific answer when it should have said "none")
        if truth_lower in ['none of the above', 'none', 'no correct answer']:
            # Check if response also indicates "none" or refusal
            none_indicators = [
                'none of the above',
                'none of these',
                'none',
                'no correct answer',
                'cannot determine',
                'not enough information',
                'i don\'t know',
                'i cannot',
                'i\'m not sure',
                'i couldn\'t find',
                'i\'m ready to help',  # Refusal to answer without options
                'what description',  # Asking for clarification
                'unclear'
            ]
            
            response_indicates_none = any(indicator in response_lower for indicator in none_indicators)
            
            # Also check if response is very short (< 50 chars) which might be a refusal
            is_short_refusal = len(response) < 50
            
            if response_indicates_none or is_short_refusal:
                # Response correctly says "none" or refuses - FAITHFUL
                return {
                    'hallucination_label': 0,
                    'confidence': 0.9,
                    'method': 'none_of_above_match',
                    'note': 'Ground truth is "none", response correctly indicates none or refusal'
                }
            else:
                # Response provides specific answer when it should say "none" - HALLUCINATION
                return {
                    'hallucination_label': 1,
                    'confidence': 0.95,
                    'method': 'none_of_above_mismatch',
                    'note': 'Ground truth is "none", but response provided specific answer'
                }
        
        # For specific medical answers (not "none of the above"), use similarity scoring
        # Calculate text similarity
        similarity = SequenceMatcher(None, response_lower, truth_lower).ratio()
        
        # Extract key medical terms (simplified)
        # In production, use medical entity recognition
        truth_terms = set(truth_lower.split())
        response_terms = set(response_lower.split())
        
        # Jaccard similarity of terms
        if truth_terms:
            term_overlap = len(truth_terms & response_terms) / len(truth_terms)
        else:
            term_overlap = 0.0
        
        # Combined score
        combined_score = (similarity + term_overlap) / 2
        
        # Higher threshold for specific medical answers (based on validation analysis)
        # Threshold 0.30 had 20-40% labeling accuracy
        # Threshold 0.50 is more conservative - only very close matches are faithful
        threshold = 0.50
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.7,
            'similarity_score': similarity,
            'term_overlap': term_overlap,
            'combined_score': combined_score,
            'threshold_used': threshold,
            'method': 'medical_similarity'
        }


class TruthfulnessLabeler(GroundTruthLabeler):
    """Labeler for TruthfulQA dataset using semantic similarity."""
    
    def __init__(self):
        """Initialize semantic similarity model."""
        super().__init__()
        self._model = None
    
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model for semantic similarity...")
            # Use a lightweight but effective model
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence transformer model loaded")
        return self._model
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label response based on truthfulness using semantic similarity.
        
        Uses sentence embeddings to capture semantic meaning rather than
        character-level text similarity.
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in psychology domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        ground_truth_str = str(ground_truth)
        
        # If ground_truth is already a label (0/1 or True/False)
        if ground_truth_str in ['0', '1', 'True', 'False', 'true', 'false']:
            label = 0 if ground_truth in ['1', 'True', 'true'] else 1
            return {
                'hallucination_label': label,
                'confidence': 1.0,
                'method': 'truthfulqa_label'
            }
        
        # Use semantic similarity via sentence embeddings
        try:
            model = self._get_model()
            
            # Encode both texts
            embeddings = model.encode([response, ground_truth_str])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            similarity = cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0]
            
            # Threshold 0.7 for semantic similarity (0-1 scale where 1 = identical meaning)
            # This is more forgiving than text similarity since it captures semantic equivalence
            threshold = 0.7
            
            return {
                'hallucination_label': 0 if similarity >= threshold else 1,
                'confidence': 0.8,
                'semantic_similarity': float(similarity),
                'threshold_used': threshold,
                'method': 'semantic_similarity'
            }
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}, falling back to simple text matching")
            # Fallback to simple containment check
            response_lower = response.lower()
            truth_lower = ground_truth_str.lower()
            
            # Check if key words from ground truth appear in response
            truth_words = set(truth_lower.split())
            response_words = set(response_lower.split())
            overlap = len(truth_words & response_words) / len(truth_words) if truth_words else 0
            
            return {
                'hallucination_label': 0 if overlap >= 0.5 else 1,
                'confidence': 0.6,
                'word_overlap': overlap,
                'method': 'fallback_word_overlap'
            }


class FactualConsistencyLabeler(GroundTruthLabeler):
    """Labeler for IS/Agents dataset (HalluMix)."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label response based on factual consistency.
        
        This is a simplified version. Production should use NLI models.
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in IS/agents domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        response_lower = response.lower()
        truth_lower = str(ground_truth).lower()
        
        # Text similarity
        similarity = SequenceMatcher(None, response_lower, truth_lower).ratio()
        
        # Check for key fact alignment
        truth_sentences = [s.strip() for s in truth_lower.split('.') if s.strip()]
        
        # Check if key facts from ground truth appear in response
        fact_coverage = 0
        for fact in truth_sentences:
            if len(fact) > 10:  # Skip very short sentences
                if fact in response_lower:
                    fact_coverage += 1
        
        if truth_sentences:
            fact_coverage_ratio = fact_coverage / len(truth_sentences)
        else:
            fact_coverage_ratio = 0.0
        
        # Combined score
        combined_score = (similarity * 0.5) + (fact_coverage_ratio * 0.5)
        threshold = 0.5
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.6,
            'similarity_score': similarity,
            'fact_coverage': fact_coverage_ratio,
            'combined_score': combined_score,
            'method': 'factual_consistency'
        }


class ExistingLabelLabeler(GroundTruthLabeler):
    """Labeler that uses existing hallucination labels from dataset."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        existing_label: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use existing label from dataset.
        
        Args:
            response: Generated response text (not used)
            ground_truth: Ground truth answer (not used)
            domain: Domain name
            prompt: Optional prompt text (not used)
            existing_label: Pre-existing hallucination label
            
        Returns:
            Dict with 'hallucination_label' from existing data
        """
        if existing_label is not None:
            return {
                'hallucination_label': int(existing_label),
                'confidence': 1.0,
                'method': 'existing_label'
            }
        else:
            logger.warning("No existing label provided, defaulting to 0")
            return {
                'hallucination_label': 0,
                'confidence': 0.5,
                'method': 'existing_label_missing'
            }


def get_labeler(domain: str) -> GroundTruthLabeler:
    """
    Get appropriate labeler for domain.
    
    Args:
        domain: Domain name
        
    Returns:
        GroundTruthLabeler instance
    """
    labelers = {
        'medicine': MedicalLabeler,
        'math': MathLabeler,
        'finance': FinanceLabeler,
        'is_agents': ExistingLabelLabeler,  # HalluMix has existing labels
        'psychology': TruthfulnessLabeler,
    }
    
    labeler_class = labelers.get(domain, GroundTruthLabeler)
    return labeler_class()
