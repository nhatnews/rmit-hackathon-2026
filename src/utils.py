"""
src/utils.py
Utilities for random seeds, 3-class Macro-F1 metrics calculation, label encoding/decoding,
and strict submission file validation for RMIT Hackathon 2026 (Challenge 1).

STRICT RULE ENFORCED:
Always validate submission files before saving:
- Columns: 'id', 'label'
- Row count matching test.csv (6,000 rows)
- No NaN / null / inf / duplicate IDs
- Values strictly in {'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}
"""

from pathlib import Path
import os
import sys
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Sequence, Union

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

from .config import (
    SUBMISSIONS_DIR,
    RANDOM_SEED,
    VALID_LABELS,
    LABEL2ID,
    ID2LABEL,
)


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


def encode_labels(y: Union[Sequence[Any], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Converts string labels ('BENIGN', 'JAILBREAK', 'PROMPT_INJECTION') to integer IDs (0, 1, 2).
    If `y` is already integer/numeric, returns it as an integer numpy array.
    """
    arr = np.asarray(y)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(int)
    try:
        return np.array([LABEL2ID[str(val).strip()] for val in arr], dtype=int)
    except KeyError as e:
        raise ValueError(
            f"Unknown label encountered during encoding: {e}. "
            f"Expected one of {VALID_LABELS}."
        ) from e


def decode_labels(y_ids: Union[Sequence[Any], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Converts integer class IDs (0, 1, 2) back to official string labels
    ('BENIGN', 'JAILBREAK', 'PROMPT_INJECTION').
    If `y_ids` already contains valid string labels, returns them as a string array.
    """
    arr = np.asarray(y_ids)
    if arr.dtype.kind in ("U", "S", "O"):
        sample_str = str(arr[0]).strip() if len(arr) > 0 else ""
        if sample_str in LABEL2ID:
            return np.array([str(val).strip() for val in arr], dtype=object)
    try:
        return np.array([ID2LABEL[int(val)] for val in arr], dtype=object)
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Invalid class index encountered during decoding: {e}. "
            f"Expected integer indices in {list(ID2LABEL.keys())}."
        ) from e


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[Any]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[Any]],
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Computes comprehensive 3-class classification metrics for RMIT Hackathon 2026.
    Primary competition metric: `f1_macro` (Macro-averaged F1 across BENIGN, JAILBREAK, PROMPT_INJECTION).
    Also reports per-class F1 scores (`f1_BENIGN`, `f1_JAILBREAK`, `f1_PROMPT_INJECTION`).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Normalize to integer IDs (0, 1, 2) if string labels are passed
    uses_official_labels = (
        (y_true_arr.dtype.kind in ("U", "S", "O") and set(np.unique(y_true_arr)).issubset(set(VALID_LABELS)))
        or (y_pred_arr.dtype.kind in ("U", "S", "O") and set(np.unique(y_pred_arr)).issubset(set(VALID_LABELS)))
        or set(np.unique(np.concatenate([y_true_arr, y_pred_arr]))).issubset({0, 1, 2})
    )

    if uses_official_labels:
        y_true_idx = encode_labels(y_true_arr)
        y_pred_idx = encode_labels(y_pred_arr)
        label_indices = [0, 1, 2]
        label_names = list(VALID_LABELS)
    else:
        y_true_idx = y_true_arr
        y_pred_idx = y_pred_arr
        label_indices = list(np.unique(np.concatenate([y_true_idx, y_pred_idx])))
        label_names = [str(lbl) for lbl in label_indices]

    f1_macro_val = float(f1_score(y_true_idx, y_pred_idx, labels=label_indices, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true_idx, y_pred_idx, labels=label_indices, average=None, zero_division=0)

    metrics: Dict[str, float] = {
        "f1_macro": f1_macro_val,
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "precision_macro": float(precision_score(y_true_idx, y_pred_idx, labels=label_indices, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_idx, y_pred_idx, labels=label_indices, average="macro", zero_division=0)),
    }

    for name, score in zip(label_names, per_class_f1):
        metrics[f"f1_{name}"] = float(score)

    # Keep f1_binary alias pointing to f1_macro for backward compatibility
    metrics["f1_binary"] = f1_macro_val

    if y_proba is not None:
        try:
            y_proba_arr = np.asarray(y_proba, dtype=float)
            if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] >= 3:
                # Normalize rows to sum to 1 for numerical safety
                row_sums = y_proba_arr.sum(axis=1, keepdims=True)
                proba_norm = np.divide(y_proba_arr, np.where(row_sums > 0, row_sums, 1.0))
                metrics["log_loss"] = float(log_loss(y_true_idx, proba_norm, labels=label_indices))
                if len(np.unique(y_true_idx)) == len(label_indices):
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true_idx, proba_norm, multi_class="ovr", average="macro", labels=label_indices)
                    )
            elif y_proba_arr.ndim == 1 or (y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == 2):
                proba_pos = y_proba_arr[:, 1] if y_proba_arr.ndim == 2 else y_proba_arr.ravel()
                metrics["roc_auc"] = float(roc_auc_score(y_true_idx, proba_pos))
                metrics["log_loss"] = float(log_loss(y_true_idx, proba_pos))
        except Exception:
            pass

    if verbose:
        cls_summary = " | ".join(
            f"F1[{name}]: {metrics.get(f'f1_{name}', 0.0):.4f}" for name in label_names
        )
        print(f"[Metrics] Macro-F1: {metrics['f1_macro']:.4f} | Acc: {metrics['accuracy']:.4f} | {cls_summary}")

    return metrics


def validate_submission(
    sub_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    id_col: str = "id",
    target_col: str = "label",
    is_probability: bool = False,
    valid_labels: Optional[Sequence[str]] = VALID_LABELS,
) -> bool:
    """
    Strict validation of submission DataFrame against RMIT Hackathon 2026 Challenge 1 rules:
    - Required columns: ['id', 'label'] (by default)
    - No NaN, null, inf, or duplicate IDs
    - Row count and ID alignment with test.csv (6,000 rows)
    - Label values strictly in {'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}
    """
    # 1. Check required columns
    missing_cols = [c for c in [id_col, target_col] if c not in sub_df.columns]
    if missing_cols:
        raise ValueError(
            f"Submission missing required columns: {missing_cols}. "
            f"Expected ['{id_col}', '{target_col}'], found: {list(sub_df.columns)}"
        )

    # 2. Check for NaN / null / inf / duplicate ids
    if sub_df[[id_col, target_col]].isna().any().any():
        nan_summary = sub_df[[id_col, target_col]].isna().sum().to_dict()
        raise ValueError(f"Submission contains NaN/null values: {nan_summary}")

    is_numeric_target = pd.api.types.is_numeric_dtype(sub_df[target_col])
    if is_numeric_target and np.isinf(sub_df[target_col]).any():
        raise ValueError(f"Submission contains infinite (inf / -inf) values in '{target_col}' column.")

    if sub_df[id_col].duplicated().any():
        raise ValueError(f"Submission has {int(sub_df[id_col].duplicated().sum())} duplicated '{id_col}' values.")

    # 3. Check against test_df if provided
    if test_df is not None:
        if len(sub_df) != len(test_df):
            raise ValueError(
                f"Row count mismatch! Submission has {len(sub_df):,} rows, "
                f"but test.csv has {len(test_df):,} rows."
            )

        if id_col in test_df.columns:
            if not (sub_df[id_col].astype(str).values == test_df[id_col].astype(str).values).all():
                raise ValueError(
                    f"Submission '{id_col}' values do not exactly match test.csv '{id_col}' values in order!"
                )

    # 4. Check target value bounds / allowed class labels
    target_vals = sub_df[target_col].values
    if is_probability:
        if not is_numeric_target:
            raise ValueError("Probability submission must have a numeric target column.")
        min_val, max_val = float(np.min(target_vals)), float(np.max(target_vals))
        if min_val < 0.0 or max_val > 1.0:
            raise ValueError(
                f"Probability submission out of bounds [0.0, 1.0]! "
                f"Min={min_val:.4f}, Max={max_val:.4f}"
            )
        if float(np.std(target_vals)) < 1e-9:
            raise ValueError("Probability submission is constant - model/pipeline bug suspected.")
    else:
        if valid_labels is not None:
            allowed_set = set(valid_labels)
            actual_set = set(pd.Series(target_vals).astype(str).str.strip().unique())
            invalid_vals = actual_set - allowed_set
            if invalid_vals:
                raise ValueError(
                    f"Submission column '{target_col}' contains invalid labels: {sorted(invalid_vals)}. "
                    f"Allowed labels are strictly: {sorted(allowed_set)}."
                )
        else:
            if is_numeric_target and not np.all(np.mod(target_vals, 1) == 0):
                raise ValueError(
                    "Label submission contains non-integer float values (looks like probabilities)."
                )

        if len(np.unique(target_vals)) < 2:
            print("[Warning] Label submission contains only a single unique class across all rows!")

    # Print class distribution summary for sanity check
    dist_counts = sub_df[target_col].value_counts()
    dist_pct = (sub_df[target_col].value_counts(normalize=True) * 100).round(2)
    dist_str = ", ".join(f"{lbl}: {dist_counts[lbl]:,} ({dist_pct[lbl]:.1f}%)" for lbl in dist_counts.index)
    print(f"[VALIDATION PASSED] Submission valid: {len(sub_df):,} rows | Columns: [{id_col}, {target_col}]")
    print(f"[CLASS DISTRIBUTION] {dist_str}")
    return True


def save_submission(
    sub_df: pd.DataFrame,
    filename: str,
    test_df: Optional[pd.DataFrame] = None,
    id_col: str = "id",
    target_col: str = "label",
    is_probability: bool = False,
    validate: bool = True,
) -> Path:
    """
    Validates and saves submission CSV (`id, label`) to submissions/ directory.
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
