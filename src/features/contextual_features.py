"""
Contextual features for HHPF.

Implements:
1. Knowledge Popularity: Entity rarity in prompts
2. Prompt Complexity: Lexical and syntactic features
"""

import numpy as np
import spacy
from typing import List, Dict, Optional, Set
import logging
from collections import Counter
import re

from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgePopularityCalculator:
    """Calculate knowledge popularity (entity rarity) in prompts."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize knowledge popularity calculator.
        
        Args:
            spacy_model: spaCy model for NER
        """
        self.config = load_config("features")['contextual_features']['knowledge_popularity']
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
            raise
        
        # Entity types to extract
        self.entity_types = self.config['entity_extraction'].get(
            'entity_types',
            ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"]
        )
        
        # Frequency data (simplified - in production, use actual Wikipedia frequencies)
        self.use_precomputed = self.config.get('frequency_source') == 'precomputed'
        
        if self.use_precomputed:
            # Load precomputed frequencies if available
            # For now, use a simple heuristic
            logger.warning("Using heuristic entity frequencies. For production, use actual Wikipedia data.")
            self.entity_frequencies = {}
        else:
            self.entity_frequencies = {}
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entities with text and label
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'text_normalized': ent.text.lower().strip()
                })
        
        return entities
    
    def estimate_entity_frequency(self, entity_text: str, entity_label: str) -> float:
        """
        Estimate entity frequency (simplified version).
        
        In production, query Wikipedia or use precomputed frequencies.
        
        Args:
            entity_text: Entity text
            entity_label: Entity type
            
        Returns:
            Estimated frequency (0-1)
        """
        entity_lower = entity_text.lower()
        
        # Check cache
        if entity_lower in self.entity_frequencies:
            return self.entity_frequencies[entity_lower]
        
        # Heuristic: Length-based frequency estimation
        # Shorter, common words tend to be more frequent
        # This is a placeholder - use real data in production
        
        # Very common entities (hardcoded for now)
        common_entities = {
            'usa', 'us', 'united states', 'america', 'europe', 'china',
            'google', 'apple', 'microsoft', 'amazon', 'facebook',
            'new york', 'london', 'paris', 'tokyo',
            'john', 'mary', 'david', 'sarah'
        }
        
        if entity_lower in common_entities:
            frequency = 0.8
        elif len(entity_text) <= 5:
            frequency = 0.6
        elif len(entity_text) <= 10:
            frequency = 0.4
        else:
            frequency = 0.2
        
        # Cache it
        self.entity_frequencies[entity_lower] = frequency
        
        return frequency
    
    def calculate_entity_rarity(self, entities: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Calculate rarity scores for entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary with rarity metrics
        """
        if not entities:
            return {
                'avg_entity_rarity': 0.0,
                'max_entity_rarity': 0.0,
                'min_entity_rarity': 0.0,
                'num_entities': 0,
                'num_rare_entities': 0
            }
        
        rarity_scores = []
        
        for entity in entities:
            frequency = self.estimate_entity_frequency(
                entity['text_normalized'],
                entity['label']
            )
            
            # Rarity = -log(frequency + smoothing)
            smoothing = self.config['rarity_calculation'].get('smoothing', 1.0)
            rarity = -np.log(frequency + smoothing)
            rarity_scores.append(rarity)
        
        rarity_scores = np.array(rarity_scores)
        
        # Aggregation
        aggregation = self.config['rarity_calculation'].get('aggregation', 'mean')
        
        if aggregation == 'mean':
            agg_rarity = np.mean(rarity_scores)
        elif aggregation == 'max':
            agg_rarity = np.max(rarity_scores)
        elif aggregation == 'min':
            agg_rarity = np.min(rarity_scores)
        else:
            agg_rarity = np.mean(rarity_scores)
        
        # Count rare entities (bottom 25% frequency)
        rare_threshold = 0.25
        num_rare = sum(1 for entity in entities 
                      if self.estimate_entity_frequency(
                          entity['text_normalized'],
                          entity['label']
                      ) < rare_threshold)
        
        return {
            'avg_entity_rarity': float(agg_rarity),
            'max_entity_rarity': float(np.max(rarity_scores)),
            'min_entity_rarity': float(np.min(rarity_scores)),
            'num_entities': len(entities),
            'num_rare_entities': num_rare,
            'entity_types': list(set(e['label'] for e in entities))
        }
    
    def calculate_knowledge_popularity(self, text: str) -> Dict[str, float]:
        """
        Calculate knowledge popularity features for text.
        
        Args:
            text: Input text (prompt)
            
        Returns:
            Dictionary with knowledge popularity features
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Calculate rarity
        rarity_features = self.calculate_entity_rarity(entities)
        
        return rarity_features


class PromptComplexityCalculator:
    """Calculate prompt complexity features."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize prompt complexity calculator.
        
        Args:
            spacy_model: spaCy model for parsing
        """
        self.config = load_config("features")['contextual_features']['prompt_complexity']
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found")
            raise
    
    def calculate_lexical_features(self, text: str, doc: Optional[spacy.tokens.Doc] = None) -> Dict[str, float]:
        """
        Calculate lexical complexity features.
        
        Args:
            text: Input text
            doc: Preprocessed spaCy Doc (optional)
            
        Returns:
            Dictionary with lexical features
        """
        if doc is None:
            doc = self.nlp(text)
        
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        
        if not tokens:
            return {
                'token_count': 0,
                'unique_token_ratio': 0.0,
                'avg_word_length': 0.0,
                'lexical_diversity': 0.0
            }
        
        # Token count
        token_count = len(tokens)
        
        # Unique tokens
        unique_tokens = set(token.text.lower() for token in tokens)
        unique_ratio = len(unique_tokens) / token_count if token_count > 0 else 0.0
        
        # Average word length
        word_lengths = [len(token.text) for token in tokens]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0.0
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = unique_ratio
        
        return {
            'token_count': token_count,
            'unique_token_ratio': unique_ratio,
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity
        }
    
    def calculate_syntactic_features(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """
        Calculate syntactic complexity features.
        
        Args:
            doc: spaCy Doc
            
        Returns:
            Dictionary with syntactic features
        """
        if len(doc) == 0:
            return {
                'avg_parse_depth': 0.0,
                'num_clauses': 0,
                'avg_dependency_length': 0.0
            }
        
        # Parse tree depth
        def get_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth
        
        depths = [get_depth(token) for token in doc]
        avg_depth = np.mean(depths) if depths else 0.0
        
        # Number of clauses (approximate by counting verbs)
        num_clauses = sum(1 for token in doc if token.pos_ == 'VERB')
        
        # Dependency arc length
        arc_lengths = []
        for token in doc:
            if token.head != token:
                arc_length = abs(token.i - token.head.i)
                arc_lengths.append(arc_length)
        
        avg_arc_length = np.mean(arc_lengths) if arc_lengths else 0.0
        
        return {
            'avg_parse_depth': avg_depth,
            'num_clauses': num_clauses,
            'avg_dependency_length': avg_arc_length
        }
    
    def identify_question_type(self, text: str) -> Dict[str, int]:
        """
        Identify question type.
        
        Args:
            text: Input text
            
        Returns:
            One-hot encoded question type
        """
        text_lower = text.lower()
        
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        
        # Find which question word appears first
        first_question = None
        min_pos = len(text)
        
        for qword in question_words:
            pos = text_lower.find(qword)
            if pos != -1 and pos < min_pos:
                min_pos = pos
                first_question = qword
        
        # One-hot encode
        result = {f'qtype_{qword}': 0 for qword in question_words}
        result['qtype_other'] = 0
        
        if first_question:
            result[f'qtype_{first_question}'] = 1
        else:
            result['qtype_other'] = 1
        
        return result
    
    def calculate_prompt_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculate all prompt complexity features.
        
        Args:
            text: Input text (prompt)
            
        Returns:
            Dictionary with all complexity features
        """
        # Parse text
        doc = self.nlp(text)
        
        features = {}
        
        # Lexical features
        if 'lexical_features' in self.config:
            lexical = self.calculate_lexical_features(text, doc)
            features.update(lexical)
        
        # Syntactic features
        if 'syntactic_features' in self.config:
            syntactic = self.calculate_syntactic_features(doc)
            features.update(syntactic)
        
        # Question type
        if self.config.get('question_type', {}).get('enabled', True):
            question_type = self.identify_question_type(text)
            features.update(question_type)
        
        return features


def extract_contextual_features(prompt: str) -> Dict[str, float]:
    """
    Extract all contextual features from a prompt.
    
    Args:
        prompt: Input prompt text
        
    Returns:
        Dictionary with all contextual features
    """
    features = {}
    
    # Knowledge popularity
    knowledge_calc = KnowledgePopularityCalculator()
    knowledge_features = knowledge_calc.calculate_knowledge_popularity(prompt)
    features.update(knowledge_features)
    
    # Prompt complexity
    complexity_calc = PromptComplexityCalculator()
    complexity_features = complexity_calc.calculate_prompt_complexity(prompt)
    features.update(complexity_features)
    
    return features
