"""
src/config.py
Configuration, paths, constants, and environment secret handlers.
"""

from pathlib import Path
import os
# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
ENV_FILE = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
    else:
        load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not installed
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


# Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, NOTEBOOKS_DIR, SUBMISSIONS_DIR, DOCS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

def get_api_key(service_name: str) -> str:
    """
    Safely retrieves an API key from environment variables.
    Checks Kaggle Secrets fallback if running inside Kaggle Notebook.
    """
    key_name = service_name.upper()
    if not key_name.endswith("_KEY") and not key_name.endswith("_TOKEN"):
        key_name = f"{key_name}_API_KEY"
    
    val = os.getenv(key_name)
    if val:
        return val

    # Kaggle Secrets fallback
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(key_name)
    except Exception:
        pass

    return ""
