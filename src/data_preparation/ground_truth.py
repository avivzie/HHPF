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
        domain: str
    ) -> Dict[str, Any]:
        """
        Label a response as hallucinated or faithful.
        
        Args:
            response: Generated response text
            ground_truth: Ground truth answer
            domain: Domain name
            
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
        domain: str
    ) -> Dict[str, Any]:
        """
        Label math response based on numerical accuracy.
        
        Returns:
            Dict with 'hallucination_label' (1 if wrong, 0 if correct) and metadata
        """
        response_value = self.extract_numerical_value(response)
        truth_value = self.extract_numerical_value(ground_truth)
        
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
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Label finance response with numerical tolerance.
        
        Returns:
            Dict with 'hallucination_label' and metadata
        """
        # Get tolerance from config
        dataset_config = self.datasets_config.get(domain, {})
        tolerance_pct = dataset_config.get('tolerance', 0.01)  # Default 1%
        
        # Try numerical comparison first
        math_labeler = MathLabeler()
        response_value = math_labeler.extract_numerical_value(response)
        truth_value = math_labeler.extract_numerical_value(ground_truth)
        
        if response_value is not None and truth_value is not None:
            # Check with percentage tolerance
            if truth_value == 0:
                tolerance_abs = 1e-6
            else:
                tolerance_abs = abs(truth_value * tolerance_pct)
            
            matches = abs(response_value - truth_value) <= tolerance_abs
            
            return {
                'hallucination_label': 0 if matches else 1,
                'confidence': 1.0,
                'extracted_response': response_value,
                'extracted_truth': truth_value,
                'tolerance_used': tolerance_abs,
                'method': 'numerical_tolerance'
            }
        
        # Fallback to text similarity
        similarity = SequenceMatcher(None, response.lower(), ground_truth.lower()).ratio()
        threshold = 0.7
        
        return {
            'hallucination_label': 0 if similarity >= threshold else 1,
            'confidence': 0.7,
            'similarity_score': similarity,
            'method': 'text_similarity'
        }


class MedicalLabeler(GroundTruthLabeler):
    """Labeler for medical questions (Med-HALT)."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Label medical response using text similarity and key term matching.
        
        For research validity, this should ideally use medical NLI or expert validation.
        """
        response_lower = response.lower()
        truth_lower = ground_truth.lower()
        
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
        threshold = 0.6
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.8,
            'similarity_score': similarity,
            'term_overlap': term_overlap,
            'combined_score': combined_score,
            'method': 'medical_similarity'
        }


class TruthfulnessLabeler(GroundTruthLabeler):
    """Labeler for TruthfulQA dataset."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Label response based on truthfulness.
        
        TruthfulQA may already have labels in the dataset.
        """
        # If ground_truth is already a label (0/1 or True/False)
        if ground_truth in ['0', '1', 'True', 'False', 'true', 'false']:
            label = 0 if ground_truth in ['1', 'True', 'true'] else 1
            return {
                'hallucination_label': label,
                'confidence': 1.0,
                'method': 'truthfulqa_label'
            }
        
        # Otherwise, use text similarity
        similarity = SequenceMatcher(None, response.lower(), ground_truth.lower()).ratio()
        threshold = 0.6
        
        return {
            'hallucination_label': 0 if similarity >= threshold else 1,
            'confidence': 0.7,
            'similarity_score': similarity,
            'method': 'text_similarity'
        }


class FactualConsistencyLabeler(GroundTruthLabeler):
    """Labeler for IS/Agents dataset (HalluMix)."""
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Label response based on factual consistency.
        
        This is a simplified version. Production should use NLI models.
        """
        response_lower = response.lower()
        truth_lower = ground_truth.lower()
        
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
        existing_label: int = None
    ) -> Dict[str, Any]:
        """
        Use existing label from dataset.
        
        Args:
            response: Generated response text (not used)
            ground_truth: Ground truth answer (not used)
            domain: Domain name
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
