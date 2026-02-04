"""
Utility functions for HHPF.
"""

import os
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        "together_api_key": os.getenv("TOGETHER_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Any, filepath: str):
    """Save data as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """Save data as pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_cache_path(cache_type: str, identifier: str) -> Path:
    """
    Get path for cached data.
    
    Args:
        cache_type: Type of cache (responses, features, etc.)
        identifier: Unique identifier for cached item
        
    Returns:
        Path to cache file
    """
    cache_dir = ensure_dir(f"cache/{cache_type}")
    return cache_dir / f"{identifier}.pkl"


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are configured.
    
    Returns:
        Dictionary indicating which API keys are available
    """
    env_vars = load_env()
    
    return {
        "together": bool(env_vars.get("together_api_key")),
        "groq": bool(env_vars.get("groq_api_key")),
        "openai": bool(env_vars.get("openai_api_key")),
    }
