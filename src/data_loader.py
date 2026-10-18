"""
src/data_loader.py
Data loading, safe text cleaning, and feature engineering.

STRICT RULES:
1. data/raw/ is strictly READ-ONLY. Never overwrite files in data/raw/.
2. clean_text_safety MUST NEVER strip Vietnamese/indigenous Unicode or adversarial
   punctuation blindly (do NOT use [^a-zA-Z0-9] regex).
"""

from pathlib import Path
import unicodedata
import re
import pandas as pd
from typing import Optional, Union

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(filename: str = "train.csv") -> pd.DataFrame:
    """
    Loads raw data file from data/raw/ directory in read-only mode.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {file_path}.\n"
            f"Please download the dataset from Kaggle into {RAW_DATA_DIR}/"
        )
    
    # Read CSV
    df = pd.read_csv(file_path)
    print(f"[DataLoader] Loaded {len(df):,} rows from {file_path.name}")
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """
    Saves cleaned or feature-engineered dataset into data/processed/.
    NEVER writes to data/raw/.
    """
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"[DataLoader] Saved processed data ({len(df):,} rows) to: {output_path}")
    return output_path


def clean_text_safety(
    text: Union[str, float, None],
    normalize_nfc: bool = True,
    strip_null_bytes: bool = True,
    preserve_case: bool = False
) -> str:
    """
    Safe text normalization for Security & Low-Resource Language tasks.

    CRITICAL RULES ENFORCED:
    - Normalizes Unicode to NFC (standard for Vietnamese diacritics).
    - NEVER strips Vietnamese tones or indigenous scripts (e.g. Malay, Thai, etc.).
    - NEVER strips punctuation blindly, because punctuation is an essential signal
      in Prompt Injections, Jailbreak triggers, and Delimiter escapes.
    """
    if pd.isna(text) or text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization (NFC ensures single-codepoint composite characters)
    if normalize_nfc:
        text = unicodedata.normalize("NFC", text)

    # Strip harmful null bytes or zero-width spaces that might crash parsers
    # while preserving other characters
    if strip_null_bytes:
        text = text.replace("\x00", "")

    # Normalize excessive carriage returns but preserve line structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Optionally lowercase
    if not preserve_case:
        text = text.lower()

    # Trim leading/trailing whitespace
    return text.strip()


def extract_adversarial_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Extracts lightweight structural & adversarial features from text.
    These features capture obfuscation, length anomalies, and injection syntax.
    """
    feats = pd.DataFrame(index=df.index)
    series = df[text_col].fillna("").astype(str)

    # 1. Length & Count features
    feats["char_len"] = series.apply(len)
    feats["word_count"] = series.apply(lambda s: len(s.split()))
    feats["avg_word_len"] = feats["char_len"] / (feats["word_count"] + 1)
    feats["line_count"] = series.apply(lambda s: s.count("\n") + 1)

    # 2. Case & Obfuscation Signals
    feats["upper_char_count"] = series.apply(lambda s: sum(1 for c in s if c.isupper()))
    feats["upper_ratio"] = feats["upper_char_count"] / (feats["char_len"] + 1)

    # 3. Special Punctuation & Delimiters (Common in Prompt Injection)
    delimiters = {
        "code_block": "```",
        "horizontal_rule": "---",
        "markdown_header": "###",
        "chat_template_open": "<|",
        "chat_template_close": "|>",
        "curly_brace": "{",
        "square_bracket": "[",
        "latex_math": "$$",
        "backslash": "\\",
    }
    for name, delim in delimiters.items():
        feats[f"delim_count_{name}"] = series.apply(lambda s: s.count(delim))

    # 4. Non-ASCII Ratio (Multilingual, Vietnamese diacritics, or Unicode obfuscation)
    feats["non_ascii_count"] = series.apply(lambda s: sum(1 for c in s if ord(c) > 127))
    feats["non_ascii_ratio"] = feats["non_ascii_count"] / (feats["char_len"] + 1)

    # 5. Digit & Symbol Ratio
    feats["digit_count"] = series.apply(lambda s: sum(1 for c in s if c.isdigit()))
    feats["symbol_count"] = series.apply(lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace()))
    feats["symbol_ratio"] = feats["symbol_count"] / (feats["char_len"] + 1)

    return feats
