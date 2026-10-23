import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_df = pd.read_csv("data/raw/train.csv")
y_true = train_df["label"].map({"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}).values

oof_v1 = np.load("Rembert_output/oof_probs_rembert.npy")
oof_v2 = np.load("Rempert_output_v2/oof_probs_rembert.npy")

print("--- MULTI-CHECKPOINT BLEND EXPLORATION ---")
for w2 in [0.4, 0.5, 0.6, 0.7]:
    w1 = 1.0 - w2
    oof_blend = w1 * oof_v1 + w2 * oof_v2
    score = f1_score(y_true, oof_blend.argmax(axis=1), average="macro")
    errs = (oof_blend.argmax(axis=1) != y_true).sum()
    print(f"Weight V1={w1:.1f} + V2={w2:.1f}: Base Argmax F1 = {score:.5f} | Errs = {errs}")

def opt_mult(probs, y):
    def loss(m):
        scaled = probs * np.array(m)
        return -f1_score(y, scaled.argmax(axis=1), average="macro")
    res = minimize(loss, [1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3})
    return res.x, -res.fun

best_w = 0.5
oof_10cp = (1.0 - best_w) * oof_v1 + best_w * oof_v2
m_glob, f1_glob = opt_mult(oof_10cp, y_true)
print(f"\n10-Checkpoint Global Tuned F1: {f1_glob:.5f} | Multipliers: {np.round(m_glob, 3)}")

total_preds = np.zeros(len(y_true), dtype=int)
for lang in ["pt", "fr", "it", "en"]:
    mask = (train_df["language"] == lang).values
    m_l, f1_l = opt_mult(oof_10cp[mask], y_true[mask])
    scaled = oof_10cp[mask] * m_l
    total_preds[mask] = scaled.argmax(axis=1)
    errs = (total_preds[mask] != y_true[mask]).sum()
    print(f"Lang {lang}: Tuned F1 = {f1_l:.5f} ({errs} errs) | Mult = {np.round(m_l, 3)}")

f1_final = f1_score(y_true, total_preds, average="macro")
print(f"\n===> 10-Checkpoint Per-Language Tuned F1: {f1_final:.5f} (Total errs: {(total_preds != y_true).sum()})")
