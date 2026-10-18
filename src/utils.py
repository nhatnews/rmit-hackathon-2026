"""
src/utils.py
Utilities for random seeds, metrics calculation, and submission file validation.

STRICT RULE ENFORCED:
Always validate submission files before saving:
- Columns: 'id', 'target'
- Row count matching test.csv
- No NaN / null / inf
- Value range checking (probability vs label)
"""

from pathlib import Path
import os
import sys
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from .config import SUBMISSIONS_DIR, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """
    Computes a comprehensive set of classification metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    if y_proba is not None:
        try:
            # If 2D probability array is passed, extract positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba.ravel()

            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            metrics["log_loss"] = float(log_loss(y_true, proba_pos))
        except Exception as e:
            metrics["roc_auc_error"] = str(e)

    return metrics


def validate_submission(
    sub_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    id_col: str = "id",
    target_col: str = "target",
    is_probability: bool = False,
) -> bool:
    """
    Strict validation of submission DataFrame against competition requirements.
    Raises ValueError with actionable error message on failure.
    """
    # 1. Check required columns
    missing_cols = [c for c in [id_col, target_col] if c not in sub_df.columns]
    if missing_cols:
        raise ValueError(f"Submission missing required columns: {missing_cols}. Current columns: {list(sub_df.columns)}")

    # 2. Check for NaN / null / inf
    if sub_df.isna().any().any():
        nan_summary = sub_df.isna().sum().to_dict()
        raise ValueError(f"Submission contains NaN values: {nan_summary}")

    if np.isinf(sub_df[target_col]).any():
        raise ValueError("Submission contains infinite (inf / -inf) values in target column.")

    # 3. Check against test_df if provided
    if test_df is not None:
        if len(sub_df) != len(test_df):
            raise ValueError(
                f"Row count mismatch! Submission has {len(sub_df):,} rows, "
                f"but test.csv has {len(test_df):,} rows."
            )
        
        if id_col in test_df.columns:
            if not (sub_df[id_col].values == test_df[id_col].values).all():
                raise ValueError(f"Submission '{id_col}' values do not exactly match test.csv '{id_col}' values in order!")

    # 4. Check target value bounds
    target_vals = sub_df[target_col].values
    if is_probability:
        min_val, max_val = float(np.min(target_vals)), float(np.max(target_vals))
        if min_val < 0.0 or max_val > 1.0:
            raise ValueError(
                f"Probability submission out of bounds [0.0, 1.0]! "
                f"Min={min_val:.4f}, Max={max_val:.4f}"
            )
    else:
        unique_vals = set(np.unique(target_vals))
        # Ensure target values are valid labels
        if not unique_vals.issubset({0, 1}) and not all(isinstance(v, (int, np.integer)) for v in unique_vals):
            print(f"[Warning] Target contains non-binary discrete values: {unique_vals}. Verify if problem is multiclass.")

    print(f"[VALIDATION PASSED] Submission valid: {len(sub_df):,} rows, columns: [{id_col}, {target_col}]")
    return True


def save_submission(
    sub_df: pd.DataFrame,
    filename: str,
    test_df: Optional[pd.DataFrame] = None,
    id_col: str = "id",
    target_col: str = "target",
    is_probability: bool = False,
    validate: bool = True,
) -> Path:
    """
    Validates and saves submission CSV to submissions/ directory.
    """
    if validate:
        validate_submission(
            sub_df=sub_df,
            test_df=test_df,
            id_col=id_col,
            target_col=target_col,
            is_probability=is_probability,
        )

    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    out_path = SUBMISSIONS_DIR / filename
    sub_df[[id_col, target_col]].to_csv(out_path, index=False)
    print(f"[SUBMISSION SAVED] Saved to: {out_path}")
    print(f"Sample preview (first 5 rows):\n{sub_df[[id_col, target_col]].head()}\n")
    return out_path
