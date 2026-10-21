"""
src/blend.py
Blending helpers for combining OOF/test probabilities from different tracks.

Which method? ROC-AUC -> "rank". Log-loss -> "logit". F1/Accuracy -> "mean" then best_threshold().
"""

import itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score


def _rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(preds, weights, method: str = "rank") -> np.ndarray:
    """Weighted blend of 1D prediction arrays. Output is always within [0, 1]."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    if method == "rank":
        return np.column_stack([_rank(p) for p in preds]) @ w
    if method == "logit":
        z = np.column_stack([_logit(p) for p in preds]) @ w
        return 1 / (1 + np.exp(-z))
    return np.column_stack(preds) @ w


def search_weights(oofs, y, metric_fn, method: str = "rank", step: float = 0.05):
    """Grid search (non-negative, sum to 1). Returns (best_score, weights). Fewer overfit risks than Optuna."""
    grid = np.arange(0, 1 + 1e-9, step)
    best = (-np.inf, None)
    for w in itertools.product(grid, repeat=len(oofs) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = (*w, 1 - sum(w))
        score = metric_fn(y, blend(oofs, w, method))
        if score > best[0]:
            best = (score, w)
    return best


def best_threshold(y, p):
    """Threshold maximising F1 on OOF predictions. Returns (threshold, f1)."""
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    i = int(np.argmax(scores))
    return float(ts[i]), float(scores[i])


def corr_report(oofs, names=None):
    """Prints OOF correlation matrix; > 0.98 means the model adds almost nothing to the blend."""
    import pandas as pd
    names = names or [f"m{i}" for i in range(len(oofs))]
    c = pd.DataFrame(dict(zip(names, oofs))).corr(method="spearman")
    print(c.round(3))
    return c
