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
import string

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
    """Labeler for medical questions (Med-HALT) using MCQ exact-match."""
    
    def extract_selected_option(self, response: str) -> Optional[str]:
        """
        Extract the option letter (A/B/C/D) selected by the LLM.
        
        Args:
            response: LLM response text
            
        Returns:
            Option letter (A/B/C/D) or None if cannot parse
        """
        response = response.strip()
        
        # Check first few characters for single letter answer
        if len(response) > 0 and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
        
        # Check for patterns like "Answer: A" or "The answer is B"
        patterns = [
            r'\b([ABCD])\b',  # Standalone letter
            r'answer[:\s]+([ABCD])\b',  # "Answer: A" or "Answer A"
            r'option[:\s]+([ABCD])\b',  # "Option: B"
            r'choice[:\s]+([ABCD])\b',  # "Choice: C"
            r'^([ABCD])\)',  # "A)" at start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
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
        Label medical MCQ response using exact option matching.
        
        For MCQ format, the LLM selects an option (A/B/C/D) and we compare
        against the correct option index for exact-match labeling.
        """
        # Handle NULL/NaN ground truth
        if pd.isna(ground_truth) or ground_truth is None or ground_truth == '':
            logger.warning(f"NULL ground truth in medical domain - cannot label, marking as hallucination")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_ground_truth',
                'note': 'No ground truth available for comparison'
            }
        
        # Get correct option index from kwargs
        correct_index = kwargs.get('correct_index')
        
        if correct_index is not None and not pd.isna(correct_index):
            # MCQ exact-match labeling
            selected_option = self.extract_selected_option(response)
            
            if selected_option is None:
                # Could not parse LLM's selection - mark as hallucination
                logger.warning(f"Could not parse option from response: {response[:100]}")
                return {
                    'hallucination_label': 1,
                    'confidence': 0.5,
                    'method': 'mcq_parse_failure',
                    'note': 'Could not parse option letter from response',
                    'response_preview': response[:100]
                }
            
            # Convert letter to index (A=0, B=1, C=2, D=3)
            selected_index = ord(selected_option) - ord('A')
            correct_idx = int(correct_index)
            
            # Exact match
            is_correct = (selected_index == correct_idx)
            
            return {
                'hallucination_label': 0 if is_correct else 1,
                'confidence': 1.0,
                'selected_option': selected_option,
                'selected_index': selected_index,
                'correct_index': correct_idx,
                'method': 'mcq_exact_match'
            }
        
        # Fallback: free-text similarity-based labeling (for backward compatibility)
        # This path shouldn't be reached with the new MCQ format
        logger.warning("MCQ exact-match not available, falling back to similarity-based labeling")
        
        response_lower = response.lower()
        truth_lower = str(ground_truth).lower()
        
        # Helper: strip punctuation for clean term matching
        def strip_punct(text: str) -> str:
            return re.sub(r'[^\w\s]', '', text)
        
        truth_clean = strip_punct(truth_lower)
        response_clean = strip_punct(response_lower)
        
        # Containment check
        truth_word_count = len(truth_clean.split())
        gt_contained = truth_clean.strip() in response_clean
        
        if gt_contained and truth_word_count <= 5:
            return {
                'hallucination_label': 0,
                'confidence': 0.9,
                'gt_contained': True,
                'truth_word_count': truth_word_count,
                'method': 'medical_containment_fallback'
            }
        
        # Term overlap with punctuation stripped
        truth_terms = set(truth_clean.split())
        response_terms = set(response_clean.split())
        
        if truth_terms:
            term_overlap = len(truth_terms & response_terms) / len(truth_terms)
        else:
            term_overlap = 0.0
        
        # Calculate text similarity
        similarity = SequenceMatcher(None, response_lower, truth_lower).ratio()
        
        # Combined score
        combined_score = (similarity + term_overlap) / 2
        threshold = 0.30
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.7,
            'similarity_score': similarity,
            'term_overlap': term_overlap,
            'combined_score': combined_score,
            'threshold_used': threshold,
            'method': 'medical_similarity_fallback'
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
    """
    Labeler for IS/Agents dataset (HalluMix) using document-grounded comparison.
    
    Compares the LLM's response against the source documents to determine
    whether the response is factually consistent with the provided evidence.
    The 'documents' column from HalluMix serves as the ground truth.
    """
    
    def __init__(self):
        """Initialize with optional semantic similarity model."""
        super().__init__()
        self._model = None
    
    def _get_model(self):
        """Lazy load sentence transformer model for semantic similarity."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model for document-grounded labeling...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence transformer model loaded")
        return self._model
    
    def label_response(
        self, 
        response: str, 
        ground_truth: str, 
        domain: str,
        prompt: str = None,
        documents: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Label response based on factual consistency with source documents.
        
        Uses a combination of:
        1. Semantic similarity between response and documents
        2. Key term overlap to check factual grounding
        
        Args:
            response: LLM-generated response text
            ground_truth: Original answer from dataset (used as secondary reference)
            domain: Domain name
            prompt: The question that was asked
            documents: Source documents/context (primary ground truth for HalluMix)
        """
        # Determine which text to compare against
        # Prefer documents (source of truth), fall back to ground_truth
        reference_text = documents if documents and not pd.isna(documents) else ground_truth
        
        if pd.isna(reference_text) or reference_text is None or reference_text == '':
            logger.warning(f"No documents or ground truth in IS/agents domain - cannot label")
            return {
                'hallucination_label': 1,
                'confidence': 0.0,
                'method': 'null_reference',
                'note': 'No documents or ground truth available for comparison'
            }
        
        # Clean up document text (HalluMix documents are stored as JSON-like strings)
        reference_clean = str(reference_text)
        if reference_clean.startswith('[') and reference_clean.endswith(']'):
            # Strip list brackets and quotes
            reference_clean = reference_clean[1:-1].strip()
            if reference_clean.startswith('"') or reference_clean.startswith("'"):
                reference_clean = reference_clean[1:]
            if reference_clean.endswith('"') or reference_clean.endswith("'"):
                reference_clean = reference_clean[:-1]
        
        response_lower = response.lower().strip()
        reference_lower = reference_clean.lower().strip()
        
        # Method 1: Semantic similarity using sentence embeddings
        semantic_score = 0.0
        try:
            model = self._get_model()
            from sklearn.metrics.pairwise import cosine_similarity
            
            embeddings = model.encode([response_lower, reference_lower])
            semantic_score = float(cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0])
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}, using text-based methods only")
            semantic_score = None
        
        # Method 2: Key term overlap (factual grounding check)
        # Extract meaningful terms (skip short words and stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'to', 'of', 'and', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'that', 'this', 'it', 'its', 'as', 'or', 'but', 'not', 'no', 'so',
            'if', 'has', 'had', 'have', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'shall', 'there', 'their',
            'they', 'them', 'he', 'she', 'him', 'her', 'his', 'we', 'us', 'our',
            'you', 'your', 'i', 'me', 'my', 'which', 'who', 'whom', 'what',
            'when', 'where', 'how', 'than', 'then', 'also', 'just', 'about',
        }
        
        response_terms = set(response_lower.split()) - stopwords
        reference_terms = set(reference_lower.split()) - stopwords
        
        # Filter to meaningful terms (length > 2)
        response_terms = {t.strip(string.punctuation) for t in response_terms if len(t.strip(string.punctuation)) > 2}
        reference_terms = {t.strip(string.punctuation) for t in reference_terms if len(t.strip(string.punctuation)) > 2}
        
        if reference_terms:
            term_overlap = len(response_terms & reference_terms) / len(reference_terms)
        else:
            term_overlap = 0.0
        
        # Method 3: Text similarity (SequenceMatcher)
        # Truncate to avoid slow comparison on very long documents
        resp_truncated = response_lower[:1000]
        ref_truncated = reference_lower[:1000]
        text_similarity = SequenceMatcher(None, resp_truncated, ref_truncated).ratio()
        
        # Combined score
        if semantic_score is not None:
            # Weighted: semantic similarity (50%), term overlap (30%), text similarity (20%)
            combined_score = (semantic_score * 0.5) + (term_overlap * 0.3) + (text_similarity * 0.2)
        else:
            # Fallback without semantic similarity
            combined_score = (term_overlap * 0.6) + (text_similarity * 0.4)
        
        # Threshold for labeling
        # Documents are typically long context passages; a response that is grounded
        # in the documents should have reasonable overlap and semantic similarity
        threshold = 0.45
        
        return {
            'hallucination_label': 0 if combined_score >= threshold else 1,
            'confidence': 0.7,
            'semantic_similarity': semantic_score,
            'term_overlap': term_overlap,
            'text_similarity': text_similarity,
            'combined_score': combined_score,
            'threshold_used': threshold,
            'used_documents': documents is not None and not pd.isna(documents),
            'method': 'document_grounded_consistency'
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
        'is_agents': FactualConsistencyLabeler,  # Document-grounded labeling against HalluMix source docs
        'psychology': TruthfulnessLabeler,
    }
    
    labeler_class = labelers.get(domain, GroundTruthLabeler)
    return labeler_class()
