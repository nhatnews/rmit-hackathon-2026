"""
RMIT Hackathon 2026 - Source Package
Challenge 1: Multilingual Prompt-Attack Classification (3-Class: BENIGN, JAILBREAK, PROMPT_INJECTION)
Languages: Portuguese (pt), French (fr), Italian (it), English (en)
"""

from .config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SUBMISSIONS_DIR,
    RANDOM_SEED,
    ID_COL,
    TEXT_COL,
    LANG_COL,
    TARGET_COL,
    VALID_LABELS,
    LABEL2ID,
    ID2LABEL,
    SUPPORTED_LANGUAGES,
    APPROVED_MODELS,
    APPROVED_KAGGLE_MODELS_URLS,
    MAX_SUBMISSIONS,
    is_approved_model,
)
from .data_loader import (
    load_raw_data,
    save_processed_data,
    clean_text_safety,
    extract_adversarial_features,
)
from .models import (
    build_baseline_pipeline,
    BaselinePipeline,
    MulticlassSafetyPipeline,
    MultilingualTextPreprocessor,
    evaluate_cv,
)
from .utils import (
    set_seed,
    encode_labels,
    decode_labels,
    validate_submission,
    save_submission,
    compute_metrics,
)
from .folds import make_folds, load_folds
from .blend import (
    blend,
    apply_multipliers,
    proba_to_labels,
    optimize_class_multipliers,
    optimize_per_language_multipliers,
    proba_to_labels_per_language,
    search_weights,
    best_threshold,
    corr_report,
)

__all__ = [
    "BASE_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SUBMISSIONS_DIR",
    "RANDOM_SEED",
    "ID_COL",
    "TEXT_COL",
    "LANG_COL",
    "TARGET_COL",
    "VALID_LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "SUPPORTED_LANGUAGES",
    "APPROVED_MODELS",
    "APPROVED_KAGGLE_MODELS_URLS",
    "MAX_SUBMISSIONS",
    "is_approved_model",
    "load_raw_data",
    "save_processed_data",
    "clean_text_safety",
    "extract_adversarial_features",
    "build_baseline_pipeline",
    "BaselinePipeline",
    "MulticlassSafetyPipeline",
    "MultilingualTextPreprocessor",
    "evaluate_cv",
    "set_seed",
    "encode_labels",
    "decode_labels",
    "validate_submission",
    "save_submission",
    "compute_metrics",
    "make_folds",
    "load_folds",
    "blend",
    "apply_multipliers",
    "proba_to_labels",
    "optimize_class_multipliers",
    "optimize_per_language_multipliers",
    "proba_to_labels_per_language",
    "search_weights",
    "best_threshold",
    "corr_report",
]
