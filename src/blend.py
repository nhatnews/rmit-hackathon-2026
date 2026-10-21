"""
src/blend.py
Post-processing and K-Fold blending helpers for 3-Class Probability Matrices (N x 3)
in RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification.

CRITICAL COMPETITION RULE:
- BTC mandates building the solution on strictly ONE approved model.
- Multi-model ensembling / blending across different backbones (e.g. DeBERTa + XLM-R) is PROHIBITED.
- Blending functions here are intended for averaging 5-Fold checkpoints of the SAME single backbone,
  or post-processing class probability multipliers (m_0, m_1, m_2) to optimize Macro-F1.

Key capabilities:
1. Averaging 3-class probability matrices (N x 3) across folds of the same model via
   weighted mean ("mean"), log-odds/softmax ("logit"), or per-class rank ("rank").
2. Optimizing class probability multipliers (m_0, m_1, m_2) globally or per-language
   (`pt, fr, it, en`) on Out-Of-Fold (OOF) predictions to maximize Macro-averaged F1.
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from .config import VALID_LABELS, SUPPORTED_LANGUAGES
from .utils import encode_labels, decode_labels


def _rank_1d(p: np.ndarray) -> np.ndarray:
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def _rank(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        return _rank_1d(arr)
    ranked = np.column_stack([_rank_1d(arr[:, c]) for c in range(arr.shape[1])])
    row_sums = ranked.sum(axis=1, keepdims=True)
    return np.divide(ranked, np.where(row_sums > 0, row_sums, 1.0))


def _logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p_clip / (1 - p_clip))


def _softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def blend(
    preds: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    method: str = "mean",
) -> np.ndarray:
    """
    Weighted blend of model probability predictions.
    Supports 3-Class probability matrices of shape (N, 3) -> returns (N, 3) row-normalized probabilities.
    """
    if len(preds) == 0:
        raise ValueError("`preds` must contain at least one prediction array.")

    arrs = [np.asarray(p, dtype=float) for p in preds]
    if weights is None:
        w = np.full(len(arrs), 1.0 / len(arrs), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    if arrs[0].ndim == 1:
        if method == "rank":
            return np.column_stack([_rank_1d(p) for p in arrs]) @ w
        if method == "logit":
            z = np.column_stack([_logit(p) for p in arrs]) @ w
            return 1.0 / (1.0 + np.exp(-z))
        return np.column_stack(arrs) @ w

    if method == "rank":
        blended = sum(weight * _rank(p) for weight, p in zip(w, arrs))
        row_sums = blended.sum(axis=1, keepdims=True)
        return np.divide(blended, np.where(row_sums > 0, row_sums, 1.0))

    if method == "logit":
        log_probs = sum(weight * np.log(np.clip(p, 1e-6, 1.0)) for weight, p in zip(w, arrs))
        return _softmax(log_probs)

    blended = sum(weight * p for weight, p in zip(w, arrs))
    row_sums = blended.sum(axis=1, keepdims=True)
    return np.divide(blended, np.where(row_sums > 0, row_sums, 1.0))


def apply_multipliers(proba: np.ndarray, multipliers: Sequence[float]) -> np.ndarray:
    """
    Scales each class column of an (N, C) probability matrix by `multipliers` (m_0, m_1, m_2)
    and re-normalizes rows to sum to 1.0.
    """
    arr = np.asarray(proba, dtype=float)
    m = np.asarray(multipliers, dtype=float).reshape(1, -1)
    scaled = arr * m
    row_sums = scaled.sum(axis=1, keepdims=True)
    return np.divide(scaled, np.where(row_sums > 0, row_sums, 1.0))


def proba_to_labels(
    proba: np.ndarray,
    multipliers: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Converts an (N, 3) probability matrix into official string labels
    ('BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'), optionally applying class multipliers first.
    """
    arr = np.asarray(proba, dtype=float)
    if multipliers is not None:
        arr = apply_multipliers(arr, multipliers)
    pred_ids = arr.argmax(axis=1)
    return decode_labels(pred_ids)


def optimize_class_multipliers(
    y_true: Union[np.ndarray, Sequence[str], Sequence[int]],
    oof_proba: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Finds class multipliers m = [1.0, m_1, m_2] on Out-Of-Fold (OOF) 3-class probabilities
    that maximize Macro-averaged F1 Score:
        y_hat = argmax_c (m_c * P(y = c | x))
    """
    proba = np.asarray(oof_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Expected 2D probability matrix (N, C), got shape {proba.shape}.")

    y_idx = encode_labels(y_true)
    n_classes = proba.shape[1]
    label_ids = list(range(n_classes))

    base_preds = proba.argmax(axis=1)
    base_f1 = float(f1_score(y_idx, base_preds, labels=label_ids, average="macro", zero_division=0))

    best_f1 = base_f1
    best_m = np.ones(n_classes, dtype=float)

    if n_classes == 3:
        grid_vals = np.linspace(0.3, 3.0, 28)
        for m1 in grid_vals:
            for m2 in grid_vals:
                m_cand = np.array([1.0, m1, m2], dtype=float)
                preds = (proba * m_cand).argmax(axis=1)
                score = float(f1_score(y_idx, preds, labels=label_ids, average="macro", zero_division=0))
                if score > best_f1:
                    best_f1 = score
                    best_m = m_cand

    def _loss(params: np.ndarray) -> float:
        if np.any(params <= 0.05) or np.any(params > 10.0):
            return 1.0
        m_vec = np.concatenate([[1.0], params])
        preds = (proba * m_vec).argmax(axis=1)
        return -float(f1_score(y_idx, preds, labels=label_ids, average="macro", zero_division=0))

    res = minimize(
        _loss,
        x0=best_m[1:],
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5},
    )
    if -float(res.fun) > best_f1:
        best_f1 = -float(res.fun)
        best_m = np.concatenate([[1.0], res.x])

    if verbose:
        opt_preds = (proba * best_m).argmax(axis=1)
        per_cls_before = f1_score(y_idx, base_preds, labels=label_ids, average=None, zero_division=0)
        per_cls_after = f1_score(y_idx, opt_preds, labels=label_ids, average=None, zero_division=0)
        print(
            f"[Threshold Tuning] Macro-F1: {base_f1:.4f} -> {best_f1:.4f} "
            f"(+{best_f1 - base_f1:.4f}) | Multipliers: {np.round(best_m, 4).tolist()}"
        )
        if n_classes == len(VALID_LABELS):
            details = " | ".join(
                f"{lbl}: {b:.4f}->{a:.4f}"
                for lbl, b, a in zip(VALID_LABELS, per_cls_before, per_cls_after)
            )
            print(f"[Per-Class F1 Gain] {details}")

    return best_m, best_f1


def optimize_per_language_multipliers(
    y_true: Union[np.ndarray, Sequence[str], Sequence[int]],
    oof_proba: np.ndarray,
    languages: Union[np.ndarray, pd.Series, Sequence[str]],
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Optimizes 3-class multipliers (m_0, m_1, m_2) independently for each language (`pt, fr, it, en`)
    to account for language-specific error distributions (e.g. BENIGN vs JAILBREAK in `pt`).
    Returns (lang_multipliers_dict, overall_macro_f1).
    """
    proba = np.asarray(oof_proba, dtype=float)
    y_idx = encode_labels(y_true)
    lang_arr = pd.Series(languages).fillna("unknown").astype(str).str.lower().str.strip().values

    base_f1 = float(f1_score(y_idx, proba.argmax(axis=1), average="macro", zero_division=0))
    global_m, _ = optimize_class_multipliers(y_idx, proba, verbose=False)

    lang_multipliers: Dict[str, np.ndarray] = {"__default__": global_m}
    opt_preds = np.zeros(len(y_idx), dtype=int)

    for lang in np.unique(lang_arr):
        mask = lang_arr == lang
        if mask.sum() >= 30:
            m_lang, f1_lang = optimize_class_multipliers(y_idx[mask], proba[mask], verbose=False)
            lang_multipliers[lang] = m_lang
            opt_preds[mask] = (proba[mask] * m_lang).argmax(axis=1)
            if verbose:
                print(f"  [Lang={lang}] Macro-F1: {f1_lang:.4f} | Multipliers: {np.round(m_lang, 4).tolist()}")
        else:
            lang_multipliers[lang] = global_m
            opt_preds[mask] = (proba[mask] * global_m).argmax(axis=1)

    overall_f1 = float(f1_score(y_idx, opt_preds, average="macro", zero_division=0))
    if verbose:
        print(f"[Per-Language Multipliers] Overall OOF Macro-F1: {base_f1:.4f} -> {overall_f1:.4f} (+{overall_f1 - base_f1:.4f})")

    return lang_multipliers, overall_f1


def proba_to_labels_per_language(
    proba: np.ndarray,
    languages: Union[np.ndarray, pd.Series, Sequence[str]],
    lang_multipliers: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Applies language-specific multipliers from `optimize_per_language_multipliers()`
    to convert a test (N, 3) probability matrix into string labels ('BENIGN', 'JAILBREAK', 'PROMPT_INJECTION').
    """
    arr = np.asarray(proba, dtype=float)
    lang_arr = pd.Series(languages).fillna("unknown").astype(str).str.lower().str.strip().values
    default_m = lang_multipliers.get("__default__", np.ones(arr.shape[1], dtype=float))

    pred_ids = np.zeros(len(arr), dtype=int)
    for lang in np.unique(lang_arr):
        mask = lang_arr == lang
        m = np.asarray(lang_multipliers.get(lang, default_m), dtype=float)
        pred_ids[mask] = (arr[mask] * m).argmax(axis=1)

    return decode_labels(pred_ids)


def search_weights(
    oofs: Sequence[np.ndarray],
    y: Union[np.ndarray, Sequence[Any]],
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    method: str = "mean",
    step: float = 0.05,
) -> Tuple[float, Tuple[float, ...]]:
    """
    Simplex grid search (non-negative weights summing to 1) over a list of OOF probability arrays.
    Returns (best_score, best_weights).
    """
    arrs = [np.asarray(o, dtype=float) for o in oofs]
    is_multiclass = arrs[0].ndim == 2

    if is_multiclass:
        y_eval = encode_labels(y) if np.asarray(y).dtype.kind in ("U", "S", "O") else np.asarray(y, dtype=int)
        if metric_fn is None:
            def metric_fn(y_true_arr: np.ndarray, blended_proba: np.ndarray) -> float:
                return float(
                    f1_score(
                        y_true_arr,
                        blended_proba.argmax(axis=1),
                        average="macro",
                        zero_division=0,
                    )
                )
    else:
        y_eval = np.asarray(y)
        if metric_fn is None:
            def metric_fn(y_true_arr: np.ndarray, blended_proba: np.ndarray) -> float:
                return float(f1_score(y_true_arr, (blended_proba >= 0.5).astype(int), zero_division=0))

    if len(arrs) == 1:
        score = float(metric_fn(y_eval, arrs[0]))
        return score, (1.0,)

    grid = np.arange(0, 1 + 1e-9, step)
    best: Tuple[float, Optional[Tuple[float, ...]]] = (-np.inf, None)
    for w in itertools.product(grid, repeat=len(arrs) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w_full = (*w, round(1.0 - sum(w), 6))
        blended = blend(arrs, w_full, method=method)
        score = float(metric_fn(y_eval, blended))
        if score > best[0]:
            best = (score, w_full)
    return best[0], best[1] or tuple(1.0 / len(arrs) for _ in arrs)


def best_threshold(
    y: Union[np.ndarray, Sequence[Any]],
    p: np.ndarray,
) -> Tuple[Union[float, np.ndarray], float]:
    """
    Optimizes decision multipliers on 2D (N, 3) OOF predictions via `optimize_class_multipliers`.
    """
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 2:
        return optimize_class_multipliers(y, arr, verbose=False)

    ts = np.unique(np.quantile(arr, np.linspace(0.01, 0.99, 197)))
    scores = [f1_score(y, (arr >= t).astype(int), zero_division=0) for t in ts]
    i = int(np.argmax(scores))
    return float(ts[i]), float(scores[i])


def corr_report(oofs: Sequence[np.ndarray], names: Optional[Sequence[str]] = None):
    """
    Prints pairwise Spearman rank correlation matrix across models.
    Correlation > 0.98 indicates low diversity between two models.
    """
    names = list(names) if names is not None else [f"m{i}" for i in range(len(oofs))]
    flat_dict = {name: np.asarray(oof, dtype=float).ravel() for name, oof in zip(names, oofs)}
    c = pd.DataFrame(flat_dict).corr(method="spearman")
    print(c.round(3))
    return c
