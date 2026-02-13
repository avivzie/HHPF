"""
Prompt formatting for HHPF.

Creates consistent prompt templates across domains for fair comparison.
"""

from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptFormatter:
    """Format prompts consistently across domains."""
    
    def __init__(self, template_style: str = "simple"):
        """
        Initialize formatter.
        
        Args:
            template_style: Style of prompt template
                - 'simple': Direct question
                - 'instruction': Instruction-following format
                - 'chat': Chat-style format
        """
        self.template_style = template_style
    
    def format_prompt(self, question: str, domain: str, **kwargs) -> str:
        """
        Format a prompt for a specific domain.
        
        Args:
            question: Raw question text
            domain: Domain name
            **kwargs: Additional domain-specific parameters
            
        Returns:
            Formatted prompt string
        """
        domain_formatters = {
            'Medicine': self._format_medical,
            'Math': self._format_math,
            'Finance': self._format_finance,
            'IS': self._format_is_agents,
            'Psychology': self._format_psychology,
        }
        
        formatter = domain_formatters.get(domain, self._format_generic)
        return formatter(question, **kwargs)
    
    def _format_generic(self, question: str, **kwargs) -> str:
        """Generic prompt format."""
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return f"Question: {question}\n\nAnswer:"
        elif self.template_style == "chat":
            return f"<|user|>\n{question}\n<|assistant|>\n"
        else:
            return question
    
    def _format_math(self, question: str, **kwargs) -> str:
        """Format math problem prompt."""
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return (
                f"Solve the following math problem step by step.\n\n"
                f"Problem: {question}\n\n"
                f"Solution:"
            )
        else:
            return question
    
    def _format_medical(self, question: str, **kwargs) -> str:
        """Format medical question prompt with MCQ options."""
        options = kwargs.get('options')
        
        # If MCQ options provided, format as multiple choice
        if options and isinstance(options, dict):
            # Extract only numeric keys (0, 1, 2, 3) and ignore 'correct answer' key
            numeric_options = {k: v for k, v in options.items() if k.isdigit()}
            
            # Format as A/B/C/D
            prompt = "Answer the following medical question by selecting the correct option (A, B, C, or D). Reply with ONLY the letter of the correct answer.\n\n"
            prompt += f"Question: {question}\n"
            
            for key in sorted(numeric_options.keys(), key=int):
                letter = chr(65 + int(key))  # 0->A, 1->B, 2->C, 3->D
                prompt += f"{letter}) {numeric_options[key]}\n"
            
            prompt += "\nAnswer:"
            return prompt
        
        # Fallback to free-text format (for backward compatibility)
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return (
                f"Answer the following medical question accurately.\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            return question
    
    def _format_finance(self, question: str, **kwargs) -> str:
        """
        Format financial question prompt with document-grounded context.
        
        TAT-QA and other financial datasets are document-grounded. The prompt includes
        the source documents (tables + text) as context so the LLM answers based on
        provided evidence. This enables proper hallucination detection.
        """
        documents = kwargs.get('documents', '')
        
        if documents:
            return (
                f"Answer the following question based ONLY on the provided context. "
                f"Be concise and accurate.\n\n"
                f"Context: {documents}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        
        # Fallback if no documents provided (FinanceBench without context)
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return (
                f"Answer the following financial question.\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            return question
    
    def _format_is_agents(self, question: str, **kwargs) -> str:
        """
        Format IS/agents question prompt with document-grounded context.
        
        HalluMix is a document-grounded QA dataset. The prompt includes the
        source documents as context so the LLM answers based on provided evidence.
        This enables proper hallucination detection (response vs. source documents).
        """
        documents = kwargs.get('documents', '')
        
        if documents:
            return (
                f"Answer the following question based ONLY on the provided context. "
                f"Be concise and accurate.\n\n"
                f"Context: {documents}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        
        # Fallback if no documents provided
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return (
                f"Task: {question}\n\n"
                f"Response:"
            )
        else:
            return question
    
    def _format_psychology(self, question: str, **kwargs) -> str:
        """Format psychology question prompt."""
        if self.template_style == "simple":
            return question
        elif self.template_style == "instruction":
            return (
                f"Answer the following question truthfully and accurately.\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            return question
    
    @staticmethod
    def create_chat_format(prompt: str, system_message: Optional[str] = None) -> Dict[str, str]:
        """
        Create chat-format messages for API calls.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Dictionary with 'messages' list
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return {"messages": messages}
