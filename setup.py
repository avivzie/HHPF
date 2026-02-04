"""
Setup script for HHPF project.

Run this script after installing requirements to:
1. Verify all dependencies
2. Check API key configuration
3. Download required models
4. Create directory structure
"""

import sys
import subprocess
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_python_version():
    """Verify Python version is 3.9 or higher."""
    print_section("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_dependencies():
    """Verify all required packages are installed."""
    print_section("Checking Dependencies")
    
    required_packages = [
        "numpy", "pandas", "scipy",
        "xgboost", "sklearn", "torch", "transformers",
        "spacy", "nltk", "matplotlib", "seaborn",
        "yaml", "dotenv", "tqdm", "together"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == "sklearn":
                __import__("sklearn")
            elif package == "yaml":
                __import__("yaml")
            elif package == "dotenv":
                __import__("dotenv")
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def download_spacy_model():
    """Download spaCy language model."""
    print_section("Downloading spaCy Model")
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model already downloaded")
            return True
        except OSError:
            print("Downloading en_core_web_sm...")
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True)
            print("✅ spaCy model downloaded")
            return True
    except Exception as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False


def check_api_keys():
    """Check if API keys are configured."""
    print_section("Checking API Configuration")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("\nTo configure API keys:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        print("\n   cp .env.example .env")
        return False
    
    # Check if .env has actual keys
    with open(env_file, 'r') as f:
        content = f.read()
    
    has_together = "TOGETHER_API_KEY=" in content and "your_together_api_key_here" not in content
    has_groq = "GROQ_API_KEY=" in content and "your_groq_api_key_here" not in content
    
    if has_together:
        print("✅ Together AI API key configured")
    else:
        print("⚠️  Together AI API key not configured")
    
    if has_groq:
        print("✅ Groq API key configured")
    else:
        print("⚠️  Groq API key not configured")
    
    if not (has_together or has_groq):
        print("\n❌ No API keys configured")
        print("\nGet your API key from:")
        print("- Together AI: https://api.together.xyz/settings/api-keys")
        print("- Groq: https://console.groq.com/keys")
        return False
    
    print("\n✅ At least one API provider is configured")
    return True


def create_directories():
    """Create necessary directory structure."""
    print_section("Creating Directory Structure")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/features",
        "cache",
        "outputs/models",
        "outputs/figures",
        "outputs/results",
        "notebooks",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    print("\n✅ Directory structure created")
    return True


def verify_torch():
    """Verify PyTorch installation and check for MPS support."""
    print_section("Checking PyTorch Configuration")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available - GPU acceleration enabled")
        else:
            print("⚠️  MPS not available - will use CPU")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False


def main():
    """Run all setup checks."""
    print("\n" + "="*60)
    print("  HHPF Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("spaCy Model", download_spacy_model),
        ("PyTorch", verify_torch),
        ("Directories", create_directories),
        ("API Keys", check_api_keys),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print_section("Setup Summary")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("1. Place your datasets in data/raw/")
        print("2. Run: jupyter notebook notebooks/")
        print("3. Start with 01_data_exploration.ipynb")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
