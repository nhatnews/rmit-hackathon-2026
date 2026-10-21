"""
src/data_loader.py
Data loading, Unicode-safe text cleaning, and multilingual adversarial feature engineering
for RMIT Hackathon 2026 — Challenge 1 (Portuguese, French, Italian, English).

STRICT RULES:
1. data/raw/ is strictly READ-ONLY. Never overwrite files in data/raw/.
2. clean_text_safety MUST NEVER strip Romance diacritics (pt, fr, it) or adversarial
   punctuation blindly (do NOT use [^a-zA-Z0-9] regex).
"""

from pathlib import Path
import unicodedata
import re
import numpy as np
import pandas as pd
from typing import Optional, Union

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_LANGUAGES


# Multilingual regex patterns for PROMPT_INJECTION (system override / prompt leaking)
# across English (en), French (fr), Italian (it), and Portuguese (pt)
_INJECTION_PATTERNS = re.compile(
    r"(?i)\b("
    # English
    r"ignore (?:all )?(?:previous|prior|above) (?:instructions|rules|directions)|"
    r"disregard (?:all )?(?:previous|prior|above)|forget everything above|"
    r"system override|new instruction|reveal (?:your )?system prompt|"
    r"initial instructions|repeat the words above|"
    # French
    r"ignore[zr]? (?:toutes )?les instructions pr[eé]c[eé]dentes|"
    r"oubliez (?:toutes )?les r[eè]gles|ne tenez pas compte|"
    r"nouvelle instruction|invite syst[eè]me|directive syst[eè]me|"
    # Italian
    r"ignora (?:tutte )?le istruzioni precedenti|dimentica tutto ci[oò] che precede|"
    r"nuova istruzione|prompt di sistema|direttiva di sistema|istruzioni iniziali|"
    # Portuguese
    r"ignore (?:todas )?as instru[cç][oõ]es anteriores|esque[cç]a todas as regras|"
    r"desconsidere tudo acima|nova instru[cç][aã]o|prompt de sistema|instru[cç][oõ]es iniciais"
    r")\b"
)

# Multilingual regex patterns for JAILBREAK (DAN / Dev Mode / Moral bypass / Fictional framing)
_JAILBREAK_PATTERNS = re.compile(
    r"(?i)\b("
    # Personas & Modes
    r"do anything now|\bdan\b|developer mode|mode d[eé]veloppeur|"
    r"modalit[aà] sviluppatore|modo desenvolvedor|unfiltered ai|"
    r"ia sans filtre|ia senza filtri|ia sem filtros|sem censura|senza censure|"
    # Moral / Ethical bypass
    r"without (?:any )?(?:moral|ethical) (?:restrictions|limits)|"
    r"sans (?:aucune )?(?:restriction|limite) (?:morale|[eé]thique)|"
    r"senza (?:alcuna )?(?:restrizione|limite) (?:morale|etica)|"
    r"sem (?:qualquer )?(?:restri[cç][aã]o|limite) (?:moral|[eé]tica)|"
    # Hypothetical / Educational framing
    r"hypothetical (?:world|scenario)|monde hypoth[eé]tique|"
    r"mondo ipotetico|mundo hipot[eé]tico|educational purposes only|"
    r"fins purement [eé]ducatives|scopo educativo|fins educacionais"
    r")\b"
)


def load_raw_data(filename: Union[str, Path] = "train.csv") -> pd.DataFrame:
    """
    Loads raw data file from data/raw/ directory in read-only mode,
    or directly from a specific file path if provided.
    """
    path_obj = Path(filename)
    file_path = path_obj if path_obj.is_file() else (RAW_DATA_DIR / filename)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {file_path}.\n"
            f"Please download the dataset from Kaggle into {RAW_DATA_DIR}/"
        )
    
    # Read by extension. keep_default_na=False so prompts like "NA"/"null"/"None"
    # are not silently turned into NaN; ids kept as strings (e.g. "007").
    suffix = file_path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(file_path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(
            file_path, sep=sep, encoding="utf-8-sig",
            keep_default_na=False, na_values=[""], dtype={"id": str},
        )
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
    Safe text normalization for Multilingual Prompt-Attack Classification (pt, fr, it, en).

    CRITICAL RULES ENFORCED:
    - Normalizes Unicode to NFC (standard for Romance diacritics: é, è, ê, ç, ã, õ, à, ù...).
    - NEVER strips accents or non-ASCII scripts.
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

    # Strip harmful null bytes that might crash parsers while preserving other characters
    if strip_null_bytes:
        text = text.replace("\x00", "")

    # Normalize excessive carriage returns but preserve line structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Optionally lowercase
    if not preserve_case:
        text = text.lower()

    # Trim leading/trailing whitespace
    return text.strip()


def extract_adversarial_features(
    df: pd.DataFrame,
    text_col: str = "text",
    lang_col: str = "language",
) -> pd.DataFrame:
    """
    Extracts lightweight structural, long-text (up to 18k chars), language, and
    multilingual security features (en, fr, it, pt) from the dataset.
    """
    feats = pd.DataFrame(index=df.index)
    series = df[text_col].fillna("").astype(str)

    # 1. Length & Long-Prompt Anomaly Features (handles prompts up to 18k+ chars)
    feats["char_len"] = series.apply(len)
    feats["log_char_len"] = np.log1p(feats["char_len"])
    feats["word_count"] = series.apply(lambda s: len(s.split()))
    feats["avg_word_len"] = feats["char_len"] / (feats["word_count"] + 1)
    feats["line_count"] = series.apply(lambda s: s.count("\n") + 1)
    feats["is_long_prompt_1k"] = (feats["char_len"] > 1000).astype(int)
    feats["is_very_long_4k"] = (feats["char_len"] > 4000).astype(int)

    # 2. Case & Obfuscation Signals
    feats["upper_char_count"] = series.apply(lambda s: sum(1 for c in s if c.isupper()))
    feats["upper_ratio"] = feats["upper_char_count"] / (feats["char_len"] + 1)

    # 3. Special Punctuation & Delimiters (Common in Prompt Injection & Context Escape)
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

    # 4. Non-ASCII Ratio (Romance diacritics in fr/pt/it or Unicode obfuscation)
    feats["non_ascii_count"] = series.apply(lambda s: sum(1 for c in s if ord(c) > 127))
    feats["non_ascii_ratio"] = feats["non_ascii_count"] / (feats["char_len"] + 1)

    # 5. Digit, Symbol & Tail Anomaly Ratio (Injections often appended at the end of long texts)
    feats["digit_count"] = series.apply(lambda s: sum(1 for c in s if c.isdigit()))
    feats["symbol_count"] = series.apply(lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace()))
    feats["symbol_ratio"] = feats["symbol_count"] / (feats["char_len"] + 1)
    feats["tail_symbol_ratio"] = series.apply(
        lambda s: sum(1 for c in s[-500:] if not c.isalnum() and not c.isspace()) / (min(len(s), 500) + 1)
    )

    # 6. Multilingual Injection vs Jailbreak Keyword Signals (en, fr, it, pt)
    feats["inj_keyword_count"] = series.apply(lambda s: len(_INJECTION_PATTERNS.findall(s)))
    feats["jb_keyword_count"] = series.apply(lambda s: len(_JAILBREAK_PATTERNS.findall(s)))
    feats["tail_inj_keyword_count"] = series.apply(lambda s: len(_INJECTION_PATTERNS.findall(s[-600:])))

    # 7. Language Column One-Hot Encoding (pt, fr, it, en)
    if lang_col in df.columns:
        lang_series = df[lang_col].fillna("unknown").astype(str).str.lower().str.strip()
        for lang in SUPPORTED_LANGUAGES:
            feats[f"lang_{lang}"] = (lang_series == lang).astype(int)

    return feats
