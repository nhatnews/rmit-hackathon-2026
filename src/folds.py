"""
src/folds.py
Shared fold assignment so every member's OOF predictions use IDENTICAL splits
(required for leak-free 3-class blending). Near-duplicate prompts (jailbreak templates)
are grouped so they never straddle train/val.

Owner: Nhật Tân. Usage:
    folds = make_folds(train_df)            # writes data/processed/folds.csv
    oof, m, test_pred = evaluate_cv(train_df, test_df=test_df, folds=load_folds(train_df))
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import PROCESSED_DATA_DIR, RANDOM_SEED, TARGET_COL
from .data_loader import clean_text_safety

FOLDS_PATH = PROCESSED_DATA_DIR / "folds.csv"


def make_folds(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = TARGET_COL,
    id_col: str = "id",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Assigns a fold id per row (duplicates grouped) and saves data/processed/folds.csv."""
    if target_col not in df.columns and "target" in df.columns:
        target_col = "target"

    key = df[text_col].fillna("").astype(str).map(lambda s: re.sub(r"\s+", " ", clean_text_safety(s)))
    groups = pd.factorize(key)[0]
    folds = np.full(len(df), -1)
    y = df[target_col].values
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, y, groups))
    except ValueError:
        print("[Folds] StratifiedGroupKFold failed, falling back to StratifiedKFold.")
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, y))
    for f, (_, va) in enumerate(splits):
        folds[va] = f
    ids = df[id_col].astype(str) if id_col in df.columns else pd.Series(np.arange(len(df)).astype(str))
    pd.DataFrame({id_col: ids, "fold": folds}).to_csv(FOLDS_PATH, index=False)
    print(f"[Folds] Saved {FOLDS_PATH} | duplicate rows grouped: {len(df) - groups.max() - 1}")
    return folds


def load_folds(df: pd.DataFrame, id_col: str = "id") -> np.ndarray:
    """Loads saved folds aligned to df's row order; raises if ids do not match."""
    f = pd.read_csv(FOLDS_PATH, dtype={id_col: str})
    if len(f) != len(df) or (id_col in df.columns and not (f[id_col].values == df[id_col].astype(str).values).all()):
        raise ValueError("folds.csv does not match this dataframe (ids/rows differ). Re-run make_folds().")
    return f["fold"].values
