# =============================================================================
# 📊 5. (V5) TỔNG HỢP OOF CÁC FOLD ĐÃ XONG, HIỆU CHỈNH KẸP BIÊN & SINH SUBMISSION
# =============================================================================
# - Tự phát hiện các fold đã hoàn thành (oof_fold{k}.npy + val_idx_fold{k}.npy + test_fold{k}.npy).
# - Test probs = trung bình các fold đã xong (cùng 1 backbone google/rembert -> hợp lệ).
# - Hậu xử lý: 1 hệ số J toàn cục + 1 hệ số PI cho mỗi ngôn ngữ, KẸP [0.70, 1.40], co 50% về 1.0.
#   Chỉ dùng nếu Nested-CV (leave-one-fold-out) tăng >= 0.0005 so với argmax, ngược lại dùng argmax.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL2ID_L = {l: i for i, l in enumerate(LABELS)}
LANGS = ["pt", "fr", "it", "en"]
GRID = np.round(np.arange(0.70, 1.4001, 0.05), 2)
SHRINK = 0.5
MIN_GAIN = 0.0005
REF_V2_F03, REF_V3_F03 = 0.96473, 0.96498  # argmax OOF của V2/V3 trên các hàng fold 0-3


def _find(name, roots=("/kaggle/input", "data/raw", "data", ".")):
    for r in roots:
        r = Path(r)
        if (r / name).exists():
            return r / name
        if r.exists():
            hits = sorted(r.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


def macro(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def apply_mult(P, langs, mJ, mPI):
    M = np.ones_like(P)
    M[:, 1] = mJ
    for lg, v in mPI.items():
        M[langs == lg, 2] = v
    return (P * M).argmax(1)


def fit_mult(P, y, langs):
    # 1) global (mJ, mPI) trên lưới kẹp biên
    best = (macro(y, P.argmax(1)), 1.0, 1.0)
    for mJ in GRID:
        for mP in GRID:
            s = macro(y, apply_mult(P, langs, mJ, {lg: mP for lg in LANGS}))
            if s > best[0] + 1e-9:
                best = (s, mJ, mP)
    _, mJ, mP = best
    mPI = {lg: mP for lg in LANGS}
    # 2) tinh chỉnh từng ngôn ngữ (coordinate ascent 1 vòng, vẫn trong lưới kẹp)
    for lg in LANGS:
        cur = macro(y, apply_mult(P, langs, mJ, mPI))
        for v in GRID:
            trial = dict(mPI)
            trial[lg] = v
            s = macro(y, apply_mult(P, langs, mJ, trial))
            if s > cur + 1e-9:
                cur, mPI = s, trial
    # 3) co 50% về 1.0 (chống overfit, tránh thảm họa it=0.3 của V2)
    mJ_s = float(1 + SHRINK * (mJ - 1))
    mPI_s = {lg: float(1 + SHRINK * (v - 1)) for lg, v in mPI.items()}
    return mJ_s, mPI_s


def run_aggregate(out_dir="/kaggle/working", train_path=None, test_path=None, sample_folds=None):
    out_dir = Path(out_dir)
    train_df = pd.read_csv(train_path or _find("train.csv"), encoding="utf-8-sig",
                           keep_default_na=False, na_values=[""], dtype={"id": str})
    test_df = pd.read_csv(test_path or _find("test.csv"), encoding="utf-8-sig",
                          keep_default_na=False, na_values=[""], dtype={"id": str})
    y_all = train_df["label"].map(LABEL2ID_L).values
    lang_all = train_df["language"].astype(str).str.lower().values

    cand = sample_folds if sample_folds is not None else range(5)
    done = [k for k in cand if all((out_dir / f"{n}_fold{k}.npy").exists() for n in ("oof", "val_idx", "test"))]
    print(f"[*] Các fold đã hoàn thành: {done}", flush=True)
    assert len(done) >= 1, "Không có fold nào hoàn thành -> không thể sinh submission!"

    idx_list, P_list, fold_of, T_list = [], [], [], []
    for k in done:
        vi = np.load(out_dir / f"val_idx_fold{k}.npy")
        P = np.load(out_dir / f"oof_fold{k}.npy")
        T = np.load(out_dir / f"test_fold{k}.npy")
        assert P.shape == (len(vi), 3) and T.shape == (len(test_df), 3)
        idx_list.append(vi); P_list.append(P); T_list.append(T); fold_of.append(np.full(len(vi), k))
    idx = np.concatenate(idx_list)
    P = np.concatenate(P_list).astype(np.float64)
    folds = np.concatenate(fold_of)
    assert len(np.unique(idx)) == len(idx), "val_idx bị trùng giữa các fold!"
    y, langs = y_all[idx], lang_all[idx]
    T = np.mean(T_list, axis=0)

    base_pred = P.argmax(1)
    base_f1 = macro(y, base_pred)
    n_err = int((base_pred != y).sum())
    print(f"[*] OOF argmax Macro-F1 trên {len(idx):,} hàng ({len(done)} fold): {base_f1:.5f} | lỗi: {n_err}", flush=True)
    print(f"    Tham chiếu cùng hàng fold 0-3: V2={REF_V2_F03:.5f} | V3={REF_V3_F03:.5f}", flush=True)
    for lg in LANGS:
        m = langs == lg
        if m.any():
            pi = (y == 2) & m
            print(f"    {lg}: F1={macro(y[m], base_pred[m]):.4f} | PI bắt trúng {int(((base_pred == 2) & pi).sum())}/{int(pi.sum())}", flush=True)

    # Nested-CV (leave-one-fold-out) để đánh giá trung thực hậu xử lý
    if len(done) >= 2:
        nested_pred = np.empty_like(base_pred)
        for k in done:
            tr, va = folds != k, folds == k
            mJ_k, mPI_k = fit_mult(P[tr], y[tr], langs[tr])
            nested_pred[va] = apply_mult(P[va], langs[va], mJ_k, mPI_k)
        nested_f1 = macro(y, nested_pred)
    else:
        nested_f1 = -1.0
    mJ, mPI = fit_mult(P, y, langs)
    tuned_insample = macro(y, apply_mult(P, langs, mJ, mPI))
    use_tuned = nested_f1 - base_f1 >= MIN_GAIN
    print(f"[*] Nested-CV tuned F1: {nested_f1:.5f} (gain {nested_f1 - base_f1:+.5f}) | in-sample: {tuned_insample:.5f}", flush=True)
    print(f"[*] Hệ số (đã co 50%): mJ={mJ:.3f} | mPI={ {k: round(v, 3) for k, v in mPI.items()} }", flush=True)

    test_lang = test_df["language"].astype(str).str.lower().values
    pred_argmax = T.argmax(1)
    pred_tuned = apply_mult(T, test_lang, mJ, mPI)
    final = pred_tuned if use_tuned else pred_argmax
    rule = "clipped_perlang_shrunk" if use_tuned else "pure_argmax"
    print(f"[+] QUY TẮC CHỌN: {rule}", flush=True)

    def _save(pred, name):
        df = pd.DataFrame({"id": test_df["id"].astype(str), "label": [LABELS[i] for i in pred]})
        df.to_csv(out_dir / name, index=False)
        return df

    _save(pred_argmax, "submission_argmax.csv")
    _save(pred_tuned, "submission_tuned.csv")
    sub = _save(final, "submission.csv")
    np.save(out_dir / "test_probs_rembert.npy", T.astype(np.float32))
    dist = sub["label"].value_counts(normalize=True).round(4).to_dict()
    print(f"[*] Phân phối nhãn test: {dist}", flush=True)
    print(f"[*] Số nhãn khác nhau argmax vs tuned trên test: {int((pred_argmax != pred_tuned).sum())}", flush=True)

    metrics = {
        "model_name": "google/rembert", "version_tag": "V5", "folds_done": done,
        "n_oof_rows": int(len(idx)), "oof_macro_f1_base": float(base_f1), "oof_errors_base": n_err,
        "oof_macro_f1_nested_tuned": float(nested_f1), "oof_macro_f1_tuned_insample": float(tuned_insample),
        "chosen_postproc_rule": rule, "mJ": mJ, "mPI": mPI, "test_label_dist": dist,
        "oof_macro_f1_tuned": float(nested_f1 if use_tuned else base_f1),  # Cell 6 đọc key này
        "per_language_f1": {lg: float(macro(y[langs == lg], base_pred[langs == lg])) for lg in LANGS if (langs == lg).any()},
        "ref_v2_argmax_f03": REF_V2_F03, "ref_v3_argmax_f03": REF_V3_F03,
        "per_class_f1_base": dict(zip(LABELS, f1_score(y, base_pred, average=None).tolist())),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


if __name__ == "__main__" or "get_ipython" in globals():
    if globals().get("SMOKE_TEST", False):
        print("[!] SMOKE_TEST = True: bỏ qua tổng hợp.", flush=True)
    else:
        run_aggregate(os.environ.get("V5_OUT_DIR", globals().get("OUT_DIR", "/kaggle/working")))
