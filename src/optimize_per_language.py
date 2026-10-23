import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from pathlib import Path

train_df = pd.read_csv('D:/Code/rmit-hackathon-2026/data/raw/train.csv')
test_df = pd.read_csv('D:/Code/rmit-hackathon-2026/data/raw/test.csv')
folds_df = pd.read_csv('D:/Code/rmit-hackathon-2026/data/processed/folds.csv')
train_df = train_df.merge(folds_df, on='id', validate='1:1')

oof_probs = np.load('D:/Code/rmit-hackathon-2026/Rembert_output/oof_probs_rembert.npy')
test_probs = np.load('D:/Code/rmit-hackathon-2026/Rembert_output/test_probs_rembert.npy')

LABEL2ID = {'BENIGN': 0, 'JAILBREAK': 1, 'PROMPT_INJECTION': 2}
ID2LABEL = {0: 'BENIGN', 1: 'JAILBREAK', 2: 'PROMPT_INJECTION'}
y_true = train_df['label'].map(LABEL2ID).values
languages = train_df['language'].values
test_languages = test_df['language'].values

print("="*60)
print("BASE SCORES")
base_preds = oof_probs.argmax(axis=1)
print(f"Argmax OOF Macro-F1: {f1_score(y_true, base_preds, average='macro'):.5f}")

global_m = np.array([1.0, 1.4, 1.4])
v1_preds = (oof_probs * global_m).argmax(axis=1)
v1_f1 = f1_score(y_true, v1_preds, average='macro')
print(f"Version 1 Global Tuned (m=[1, 1.4, 1.4]) OOF: {v1_f1:.5f}")
print("="*60)

def optimize_multipliers_for_subset(y_sub, proba_sub):
    base_sub_preds = proba_sub.argmax(axis=1)
    best_f1 = float(f1_score(y_sub, base_sub_preds, average="macro", zero_division=0))
    best_m = np.array([1.0, 1.0, 1.0], dtype=float)

    # Grid search 35x35 for m1, m2 (range 0.4 to 3.5)
    grid_vals = np.linspace(0.4, 3.5, 35)
    for m1 in grid_vals:
        for m2 in grid_vals:
            cand = np.array([1.0, m1, m2], dtype=float)
            preds = (proba_sub * cand).argmax(axis=1)
            score = float(f1_score(y_sub, preds, average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_m = cand

    # Nelder-Mead fine-tuning
    def _loss(params):
        if np.any(params <= 0.05) or np.any(params > 10.0):
            return 1.0
        m_vec = np.concatenate([[1.0], params])
        preds = (proba_sub * m_vec).argmax(axis=1)
        return -float(f1_score(y_sub, preds, average="macro", zero_division=0))

    res = minimize(_loss, x0=best_m[1:], method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5})
    if -float(res.fun) > best_f1:
        best_f1 = -float(res.fun)
        best_m = np.concatenate([[1.0], res.x])

    return best_m, best_f1

# Optimize separately for each language!
lang_multipliers = {}
tuned_oof_preds = np.zeros(len(y_true), dtype=int)
tuned_test_preds = np.zeros(len(test_df), dtype=int)
scaled_test_probs = np.zeros_like(test_probs)

print("\nOPTIMIZING PER LANGUAGE:")
for lang in ['pt', 'fr', 'it', 'en']:
    mask = (languages == lang)
    test_mask = (test_languages == lang)
    
    y_sub = y_true[mask]
    proba_sub = oof_probs[mask]
    
    # Old score in v1
    v1_lang_f1 = f1_score(y_sub, v1_preds[mask], average='macro')
    
    best_m, new_lang_f1 = optimize_multipliers_for_subset(y_sub, proba_sub)
    lang_multipliers[lang] = best_m
    
    # Apply to OOF
    tuned_oof_preds[mask] = (proba_sub * best_m).argmax(axis=1)
    
    # Apply to Test
    proba_test_sub = test_probs[test_mask]
    scaled_sub = proba_test_sub * best_m
    scaled_test_probs[test_mask] = scaled_sub
    tuned_test_preds[test_mask] = scaled_sub.argmax(axis=1)
    
    print(f"Lang {lang:<2} ({mask.sum():>4} train samples) | Best m: {np.round(best_m, 3).tolist()} | F1: {v1_lang_f1:.5f} -> {new_lang_f1:.5f} ({'+' if new_lang_f1 >= v1_lang_f1 else ''}{new_lang_f1 - v1_lang_f1:.5f})")

overall_new_f1 = f1_score(y_true, tuned_oof_preds, average='macro')
print("="*60)
print(f"GLOBAL OOF MACRO-F1: {v1_f1:.5f} -> {overall_new_f1:.5f} (+{overall_new_f1 - v1_f1:.5f})")
print("="*60)

# Check test distribution
test_labels = [ID2LABEL[idx] for idx in tuned_test_preds]
sub_df = pd.DataFrame({'id': test_df['id'], 'label': test_labels})

print("\nNEW TEST LABEL DISTRIBUTION:")
print(sub_df['label'].value_counts())
print(sub_df['label'].value_counts(normalize=True)*100)

# Save new submission
out_path = Path('D:/Code/rmit-hackathon-2026/submissions/submission_rembert_per_lang_tuned.csv')
sub_df.to_csv(out_path, index=False)
print(f"\nSaved new submission to: {out_path}")
