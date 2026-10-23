import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")
label2id = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
id2label = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
y_true = train_df["label"].map(label2id).values

oof_v1 = np.load("Rembert_output/oof_probs_rembert.npy")
oof_v2 = np.load("Rempert_output_v2/oof_probs_rembert.npy")
test_v1 = np.load("Rembert_output/test_probs_rembert.npy")
test_v2 = np.load("Rempert_output_v2/test_probs_rembert.npy")

print("=" * 60)
print("1. BASE OOF MACRO-F1")
print("=" * 60)
score_v1 = f1_score(y_true, oof_v1.argmax(axis=1), average="macro")
score_v2 = f1_score(y_true, oof_v2.argmax(axis=1), average="macro")
print(f"V1 OOF Base (Argmax): {score_v1:.5f}")
print(f"V2 OOF Base (Argmax): {score_v2:.5f} ({score_v2 - score_v1:+.5f})")

print("\n" + "=" * 60)
print("2. MULTI-RUN CHECKPOINT AVERAGING (V1 + V2 CÙNG BACKBONE REMBERT)")
print("=" * 60)
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend_oof = (1.0 - w) * oof_v1 + w * oof_v2
    score = f1_score(y_true, blend_oof.argmax(axis=1), average="macro")
    errs = (blend_oof.argmax(axis=1) != y_true).sum()
    print(f"Weight V1={1-w:.1f} + V2={w:.1f}: Macro-F1 = {score:.5f} | Total Errors = {errs}")

print("\n" + "=" * 60)
print("3. THRESHOLD OPTIMIZATION TRÊN V2 vs V1+V2")
print("=" * 60)

def eval_multipliers(probs, y, m):
    scaled = probs * np.array(m)
    preds = scaled.argmax(axis=1)
    return f1_score(y, preds, average="macro")

def opt_mult(probs, y):
    def loss(m):
        return -eval_multipliers(probs, y, m)
    res = minimize(loss, [1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3})
    return res.x, -res.fun

m_v2, f1_v2_opt = opt_mult(oof_v2, y_true)
print(f"V2 Global Multipliers: {np.round(m_v2, 3)} -> Macro-F1 = {f1_v2_opt:.5f}")

# Check per-language on V2
print("\nV2 Per-Language Macro-F1 (Tuned vs Base):")
for lang in ["pt", "fr", "it", "en"]:
    mask = (train_df["language"] == lang).values
    f1_b = f1_score(y_true[mask], oof_v2[mask].argmax(axis=1), average="macro")
    m_l, f1_t = opt_mult(oof_v2[mask], y_true[mask])
    err_b = (oof_v2[mask].argmax(axis=1) != y_true[mask]).sum()
    scaled = oof_v2[mask] * m_l
    err_t = (scaled.argmax(axis=1) != y_true[mask]).sum()
    print(f"  Lang {lang:2s} ({mask.sum():5d} samples): Base = {f1_b:.5f} ({err_b} err) -> Tuned = {f1_t:.5f} ({err_t} err) | Mult = {np.round(m_l, 3)}")

# Test agreement
sub_v1 = pd.read_csv("Rembert_output/submission.csv")
sub_v2 = pd.read_csv("Rempert_output_v2/submission.csv")
diff_mask = sub_v1["label"] != sub_v2["label"]
diff_df = test_df[diff_mask].copy()
diff_df["v1"] = sub_v1.loc[diff_mask, "label"]
diff_df["v2"] = sub_v2.loc[diff_mask, "label"]

print("\n" + "=" * 60)
print(f"4. TEST SET FLIPPED SAMPLES (Tổng cộng {diff_mask.sum()} mẫu đổi nhãn):")
print("=" * 60)
print(diff_df["language"].value_counts().to_string())
print("\nChuyển dịch nhãn chi tiết:")
print(pd.crosstab(diff_df["v1"], diff_df["v2"]))

print("\nVí dụ các mẫu đổi nhãn tiêu biểu trên tiếng Bồ Đào Nha (pt):")
pt_diff = diff_df[diff_df["language"] == "pt"]
for idx, row in pt_diff.head(6).iterrows():
    print(f"ID {row['id']} | V1: {row['v1']} -> V2: {row['v2']}")
    snippet = row['text'][:140].replace('\n', ' ')
    print(f"  Snippet: {snippet}...\n")
