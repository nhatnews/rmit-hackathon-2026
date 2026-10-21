"""
src/config.py
Configuration, paths, constants, and environment secret handlers for
RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification.
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
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # Read-only filesystem (e.g. src uploaded as Kaggle dataset)

# Random Seed for Reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Official Competition Constants (Challenge 1: 3-Class Classification)
ID_COL = "id"
TEXT_COL = "text"
LANG_COL = "language"
TARGET_COL = "label"

VALID_LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
LABEL2ID = {label: idx for idx, label in enumerate(VALID_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(VALID_LABELS)}

SUPPORTED_LANGUAGES = ("en", "fr", "it", "pt")
MAX_SUBMISSIONS = 6

APPROVED_MODELS = (
    # Encoders (Fast & Multilingual)
    "distilbert/distilbert-base-multilingual-cased",
    "jhu-clsp/mmBERT-small",
    "jhu-clsp/mmBERT-base",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "google/rembert",
    # Encoders (Safety-tuned for Prompt Injection)
    "protectai/deberta-v3-base-prompt-injection-v2",
    "patronus-studio/wolf-defender-prompt-injection",
    # LLMs (LoRA / Fine-tuning)
    "ibm-granite/granite-3.3-2b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

APPROVED_KAGGLE_MODELS_URLS = {
    "distilbert/distilbert-base-multilingual-cased": "https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/Transformers/default",
    "jhu-clsp/mmBERT-small": "https://www.kaggle.com/models/aisuko/mmbert-small/Transformers/default",
    "jhu-clsp/mmBERT-base": "https://www.kaggle.com/models/aisuko/mmbert-base/Transformers/default",
    "FacebookAI/xlm-roberta-base": "https://www.kaggle.com/models/aisuko/xlm-roberta-base/Transformers/default",
    "FacebookAI/xlm-roberta-large": "https://www.kaggle.com/models/aisuko/xlm-roberta-large/Transformers/default",
    "google/rembert": "https://www.kaggle.com/models/aisuko/rembert/Transformers/default",
    "protectai/deberta-v3-base-prompt-injection-v2": "https://www.kaggle.com/models/aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default",
    "patronus-studio/wolf-defender-prompt-injection": "https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default",
    "ibm-granite/granite-3.3-2b-instruct": "https://www.kaggle.com/models/aisuko/granite-3.3-2b-instruct/Transformers/default",
    "mistralai/Mistral-7B-Instruct-v0.3": "https://www.kaggle.com/models/aisuko/mistral-7b-instruct-v0.3/Transformers/default",
}


def is_approved_model(model_name: str) -> bool:
    """Verifies if a given model backbone is officially approved on Kaggle."""
    name_clean = model_name.strip()
    return any(name_clean == m or name_clean.endswith(m.split("/")[-1]) for m in APPROVED_MODELS)



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
