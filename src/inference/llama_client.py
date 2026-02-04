"""
Llama-3 API client for HHPF.

Supports Together AI and Groq APIs for Llama-3 inference.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.utils import load_config, load_env, get_cache_path, save_pickle, load_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaClient:
    """Base client for Llama-3 inference."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "together",
        enable_cache: bool = True
    ):
        """
        Initialize Llama client.
        
        Args:
            model: Model name (default from config)
            provider: API provider ('together' or 'groq')
            enable_cache: Whether to cache responses
        """
        self.config = load_config("model")['llama']
        self.model = model or self.config['default_model']
        self.provider = provider
        self.enable_cache = enable_cache
        
        # Load API keys
        env = load_env()
        
        if provider == "together":
            if not TOGETHER_AVAILABLE:
                raise ImportError("together package not installed. Run: pip install together")
            
            api_key = env.get("together_api_key")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in .env")
            
            self.client = Together(api_key=api_key)
            logger.info(f"Initialized Together AI client with model: {self.model}")
            
        elif provider == "groq":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            
            api_key = env.get("groq_api_key")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            # Groq model names
            if "70b" in self.model.lower():
                self.model = "llama3-70b-8192"
            else:
                self.model = "llama3-8b-8192"
            
            logger.info(f"Initialized Groq client with model: {self.model}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'together' or 'groq'")
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for a request."""
        cache_data = {
            'prompt': prompt,
            'model': self.model,
            'provider': self.provider,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load response from cache."""
        if not self.enable_cache:
            return None
        
        try:
            cache_path = get_cache_path("responses", cache_key)
            if cache_path.exists():
                return load_pickle(cache_path)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict):
        """Save response to cache."""
        if not self.enable_cache:
            return
        
        try:
            cache_path = get_cache_path("responses", cache_key)
            save_pickle(response, cache_path)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        logprobs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from Llama-3.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            logprobs: Number of top logprobs to return
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        # Use config defaults
        max_tokens = max_tokens or self.config.get('max_tokens', 1024)
        temperature = temperature or self.config.get('temperature', 0.8)
        top_p = top_p or self.config.get('top_p', 0.95)
        
        # Check cache
        cache_key = self._get_cache_key(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs
        )
        
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.debug("Response loaded from cache")
            return cached
        
        # Rate limiting
        rate_limit_delay = self.config.get('rate_limit_delay', 0.5)
        time.sleep(rate_limit_delay)
        
        # Generate
        try:
            if self.provider == "together":
                response = self._generate_together(
                    prompt, max_tokens, temperature, top_p, logprobs, **kwargs
                )
            elif self.provider == "groq":
                response = self._generate_groq(
                    prompt, max_tokens, temperature, top_p, **kwargs
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Save to cache
            self._save_to_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_together(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        logprobs: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Together AI."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            **kwargs
        )
        
        # Parse response
        result = {
            'text': response.choices[0].text,
            'model': self.model,
            'provider': 'together',
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        }
        
        # Add logprobs if requested
        if logprobs and hasattr(response.choices[0], 'logprobs'):
            result['logprobs'] = response.choices[0].logprobs
        
        return result
    
    def _generate_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Groq."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Parse response
        result = {
            'text': response.choices[0].message.content,
            'model': self.model,
            'provider': 'groq',
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        }
        
        return result
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}")
            try:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt {i}: {e}")
                responses.append({
                    'text': '',
                    'error': str(e)
                })
        
        return responses
