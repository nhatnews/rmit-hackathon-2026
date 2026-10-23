"""
src/verify_v3_v4_gatekeeper.py
Script Kiểm Định Cổng Cuối (Gatekeeper Protocol) cho 2 lượt nộp cuối cùng (Lượt 5 & Lượt 6).
Đối chiếu toàn diện V1 (LB 0.97058) vs V2 (LB 0.97471) vs V3 (Trung Quân) vs V4 (User):
1. Điểm OOF Base (Argmax) & 5-Fold Nested-CV Macro-F1 (chống overfit cho 70% Private Test).
2. Phân rã số lỗi trên 2 tử huyệt:
   - Tử huyệt 1: Tiếng Bồ Đào Nha 'pt' (BENIGN <-> JAILBREAK).
   - Tử huyệt 2: Tiếng Pháp/Ý/Anh 'fr/it/en' (BENIGN <-> PROMPT_INJECTION).
3. Paired Bootstrap Win-Rate (500 lần lấy mẫu trên 14,000 mẫu OOF) so với V2 (LB 0.97471).
4. Kiểm định Sanity Check & Số lượng mẫu đổi nhãn (Test Flip Window) trên 6,000 mẫu test.csv.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
VALID_LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}


def find_version_dir(candidates: List[str], create_default: Optional[str] = None) -> Optional[Path]:
    for name in candidates:
        p = ROOT_DIR / name
        if p.exists():
            if (p / "oof_probs_rembert.npy").exists():
                return p
            for sub in p.rglob("oof_probs_rembert.npy"):
                return sub.parent
    if create_default is not None:
        d = ROOT_DIR / create_default
        d.mkdir(parents=True, exist_ok=True)
    return None


def fast_grid_multipliers(y_idx: np.ndarray, proba: np.ndarray) -> np.ndarray:
    base_f1 = float(f1_score(y_idx, proba.argmax(axis=1), average="macro", zero_division=0))
    best_f1 = base_f1
    best_m = np.ones(3, dtype=float)
    grid = np.linspace(0.45, 1.85, 15)
    for m1 in grid:
        for m2 in grid:
            cand = np.array([1.0, m1, m2])
            s = float(f1_score(y_idx, (proba * cand).argmax(axis=1), average="macro", zero_division=0))
            if s > best_f1 + 1e-7:
                best_f1 = s
                best_m = cand
    return best_m


def evaluate_nested_cv(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    langs: np.ndarray,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Đánh giá 5-Fold Nested-CV công bằng cho Global Multipliers và Shrunk Per-Lang Multipliers."""
    nested_glob_preds = np.zeros(len(y_true), dtype=int)
    nested_lang_preds = np.zeros(len(y_true), dtype=int)

    for val_k in range(5):
        trn = folds != val_k
        val = folds == val_k
        m_g = fast_grid_multipliers(y_true[trn], oof_probs[trn])
        nested_glob_preds[val] = (oof_probs[val] * m_g).argmax(axis=1)

        for lg in ("pt", "fr", "it", "en"):
            trn_l = trn & (langs == lg)
            val_l = val & (langs == lg)
            m_l = fast_grid_multipliers(y_true[trn_l], oof_probs[trn_l])
            m_shrunk = 0.5 * m_g + 0.5 * m_l
            nested_lang_preds[val_l] = (oof_probs[val_l] * m_shrunk).argmax(axis=1)

    f1_glob = float(f1_score(y_true, nested_glob_preds, average="macro", zero_division=0))
    f1_lang = float(f1_score(y_true, nested_lang_preds, average="macro", zero_division=0))
    return f1_glob, nested_glob_preds, f1_lang, nested_lang_preds


def count_regime_errors(y_true: np.ndarray, preds: np.ndarray, langs: np.ndarray) -> Dict[str, int]:
    """Đếm số lỗi trên Tử huyệt 1 (pt B<->J) và Tử huyệt 2 (fr/it/en B<->PI)."""
    pt_mask = langs == "pt"
    cm_pt = confusion_matrix(y_true[pt_mask], preds[pt_mask], labels=[0, 1, 2])
    pt_bj_errs = int(cm_pt[0, 1] + cm_pt[1, 0])
    pt_total_errs = int((y_true[pt_mask] != preds[pt_mask]).sum())

    other_mask = ~pt_mask
    cm_oth = confusion_matrix(y_true[other_mask], preds[other_mask], labels=[0, 1, 2])
    oth_bpi_errs = int(cm_oth[0, 2] + cm_oth[2, 0])
    oth_fp_pi = int(cm_oth[0, 2])
    oth_fn_pi = int(cm_oth[2, 0])
    total_errs = int((y_true != preds).sum())

    return {
        "total_errs": total_errs,
        "pt_total_errs": pt_total_errs,
        "pt_bj_errs": pt_bj_errs,
        "oth_bpi_errs": oth_bpi_errs,
        "oth_fp_pi": oth_fp_pi,
        "oth_fn_pi": oth_fn_pi,
    }


def paired_bootstrap_win_rate(
    y_true: np.ndarray,
    cand_preds: np.ndarray,
    ref_preds: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f_cand = f1_score(y_true[idx], cand_preds[idx], average="macro", zero_division=0)
        f_ref = f1_score(y_true[idx], ref_preds[idx], average="macro", zero_division=0)
        deltas.append(f_cand - f_ref)
    arr = np.array(deltas)
    return float(arr.mean()), float((arr > 0).mean())


def main() -> None:
    print("=" * 88)
    print("🛡️ GATEKEEPER PROTOCOL: ĐỐI CHIẾU TOÀN DIỆN REMBERT V1 vs V2 vs V3 vs V4 (LƯỢT 5 & 6)")
    print("=" * 88)

    train_df = pd.read_csv(ROOT_DIR / "data" / "raw" / "train.csv", encoding="utf-8-sig", keep_default_na=False, dtype={"id": str})
    test_df = pd.read_csv(ROOT_DIR / "data" / "raw" / "test.csv", encoding="utf-8-sig", keep_default_na=False, dtype={"id": str})
    folds_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "folds.csv", dtype={"id": str})
    train_df = train_df.merge(folds_df, on="id", how="left", validate="1:1")

    y_true = train_df["label"].map(LABEL2ID).values
    folds = train_df["fold"].astype(int).values
    langs = train_df["language"].values

    versions = {
        "V1 (LB 0.97058)": find_version_dir(["Rembert_output"]),
        "V2 (LB 0.97471)": find_version_dir(["Rempert_output_v2", "Rembert_output_v2"]),
        "V3 (Trung Quân)": find_version_dir(["Rembert_output_v3", "Rempert_output_v3"], create_default="Rembert_output_v3"),
        "V4 (User Hybrid)": find_version_dir(["Rembert_output_v4", "Rempert_output_v4"], create_default="Rembert_output_v4"),
    }

    print("\n📂 1. TRẠNG THÁI CÁC THƯ MỤC KẾT QUẢ TRONG WORKSPACE:")
    print("-" * 88)
    loaded_oof: Dict[str, np.ndarray] = {}
    loaded_test: Dict[str, np.ndarray] = {}
    loaded_sub: Dict[str, pd.DataFrame] = {}

    for label, vdir in versions.items():
        if vdir is not None:
            print(f"  ✅ {label:<18}: SẴN SÀNG ({vdir.name})")
            loaded_oof[label] = np.load(vdir / "oof_probs_rembert.npy")
            if (vdir / "test_probs_rembert.npy").exists():
                loaded_test[label] = np.load(vdir / "test_probs_rembert.npy")
            if (vdir / "submission.csv").exists():
                loaded_sub[label] = pd.read_csv(vdir / "submission.csv", dtype={"id": str})
        else:
            default_folder = "Rembert_output_v3" if "V3" in label else "Rembert_output_v4"
            print(f"  ⏳ {label:<18}: ĐANG CHỜ TẢI KẾT QUẢ VÀO '{default_folder}/' (metrics.json, oof_probs_rembert.npy, submission.csv)")

    v2_key = "V2 (LB 0.97471)"
    ref_preds = loaded_oof[v2_key].argmax(axis=1)

    print("\n📊 2. BẢNG ĐỐI CHIẾU ĐIỂM 5-FOLD OOF & PHÂN RÃ LỖI TỬ HUYỆT (14,000 MẪU):")
    print("-" * 88)
    header = (
        f"{'Phiên bản':<18} | {'Argmax F1':>9} | {'NestGlob':>9} | {'NestLang':>9} | "
        f"{'Tổng Lỗi':>8} | {'pt B<->J':>8} | {'fr/it/en PI':>11} | {'P(Thắng V2)':>11}"
    )
    print(header)
    print("-" * 88)

    for label, oof in loaded_oof.items():
        base_preds = oof.argmax(axis=1)
        base_f1 = float(f1_score(y_true, base_preds, average="macro", zero_division=0))
        f1_glob, glob_preds, f1_lang, lang_preds = evaluate_nested_cv(oof, y_true, folds, langs)
        errs = count_regime_errors(y_true, base_preds, langs)

        if label == v2_key:
            win_str = "Baseline"
        else:
            _, p_win = paired_bootstrap_win_rate(y_true, base_preds, ref_preds)
            win_str = f"{p_win * 100:5.1f}%"

        print(
            f"{label:<18} | {base_f1:9.5f} | {f1_glob:9.5f} | {f1_lang:9.5f} | "
            f"{errs['total_errs']:8d} | {errs['pt_bj_errs']:8d} | "
            f"{errs['oth_bpi_errs']:4d} (FP:{errs['oth_fp_pi']:2d}) | {win_str:>11}"
        )

    # Kiểm tra đối chiếu trên tập Test (6,000 mẫu) so với V2
    if v2_key in loaded_sub:
        sub_v2 = loaded_sub[v2_key]
        print("\n🔍 3. ĐỐI CHIẾU DỰ ĐOÁN TRÊN TEST.CSV (6,000 MẪU) SO VỚI V2 (LB 0.97471):")
        print("-" * 88)
        for label, sub_df in loaded_sub.items():
            assert len(sub_df) == 6000, f"[{label}] Sai số dòng: {len(sub_df)}"
            assert list(sub_df.columns) == ["id", "label"], f"[{label}] Sai cột: {list(sub_df.columns)}"
            assert not sub_df.isna().any().any(), f"[{label}] Có NaN!"
            assert set(sub_df["label"].unique()).issubset(set(VALID_LABELS)), f"[{label}] Nhãn lạ!"
            assert (sub_df["id"].values == test_df["id"].values).all(), f"[{label}] Thứ tự ID lệch!"

            flips = int((sub_df["label"] != sub_v2["label"]).sum())
            dist = sub_df["label"].value_counts(normalize=True) * 100
            print(
                f"  - {label:<18}: Đổi nhãn so với V2 = {flips:3d} / 6,000 mẫu | "
                f"B={dist.get('BENIGN', 0):.2f}%, J={dist.get('JAILBREAK', 0):.2f}%, PI={dist.get('PROMPT_INJECTION', 0):.2f}%"
            )

    print("\n" + "=" * 88)
    print("✅ QUY TẮC QUYẾT ĐỊNH NỘP BÀI (GATEKEEPER CHECKLIST CHO LƯỢT 5 & LƯỢT 6):")
    print("  1. Điều kiện ĐẠT (PASS) để bấm Submit trên Kaggle:")
    print("     - OOF Argmax F1 >= 0.9670 (hoặc Nested-CV F1 >= 0.9675) VÀ Tổng lỗi OOF <= 368 lỗi.")
    print("     - Giảm được lỗi ở Tử huyệt 1 (pt B<->J < 253) hoặc Tử huyệt 2 (fr/it/en FP PI < 35).")
    print("     - Xác suất thắng V2 theo Paired Bootstrap P(Thắng V2) >= 85%.")
    print("     - Số mẫu đổi nhãn trên Test so với V2 nằm trong khung vàng: 25 - 90 mẫu.")
    print("=" * 88)


if __name__ == "__main__":
    main()
