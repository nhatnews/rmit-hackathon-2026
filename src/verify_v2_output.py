"""
verify_v2_output.py
Script tự động kiểm định toàn diện kết quả đầu ra, log huấn luyện và dự đoán của RemBERT Version 2
so với RemBERT Version 1 (Personal Best LB: 0.97058).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
V1_DIR = ROOT_DIR / "Rembert_output"
# Hỗ trợ cả 2 cách đặt tên thư mục (Rempert_output_v2 hoặc Rembert_output_v2)
V2_CANDIDATES = [
    ROOT_DIR / "Rempert_output_v2",
    ROOT_DIR / "Rembert_output_v2",
]

VALID_LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}


def find_v2_dir() -> Path:
    for d in V2_CANDIDATES:
        if d.exists() and any(d.iterdir()):
            return d
    for d in V2_CANDIDATES:
        if d.exists():
            return d
    return V2_CANDIDATES[0]


def main() -> None:
    print("=" * 78)
    print("🔍 BÁO CÁO NGHIỆM THU & ĐỐI CHIẾU REMBERT VERSION 2 VS VERSION 1 (LB 0.97058)")
    print("=" * 78)

    v2_dir = find_v2_dir()
    print(f"[*] Thư mục V1 (Baseline 0.97058): {V1_DIR}")
    print(f"[*] Thư mục V2 (Cần nghiệm thu)  : {v2_dir}")

    if not v2_dir.exists() or not any(v2_dir.iterdir()):
        print(f"\n[!] CẢNH BÁO: Thư mục '{v2_dir}' hiện đang trống!")
        print("    -> Vui lòng tải các file output từ Kaggle Notebook Version 2 vào thư mục này:")
        print("       1. metrics.json")
        print("       2. submission.csv")
        print("       3. oof_probs_rembert.npy")
        print("       4. test_probs_rembert.npy")
        print("       5. Thư mục logs/ (nếu có)")
        sys.exit(1)

    # 1. Kiểm tra danh sách file trong V2
    v2_metrics_path = v2_dir / "metrics.json"
    v2_sub_path = v2_dir / "submission.csv"
    v2_oof_path = v2_dir / "oof_probs_rembert.npy"
    v2_test_path = v2_dir / "test_probs_rembert.npy"

    for p in [v2_metrics_path, v2_sub_path, v2_oof_path, v2_test_path]:
        status = "✅ CÓ" if p.exists() else "❌ THIẾU"
        print(f"    - {p.name:<24}: {status}")

    # 2. Đọc dữ liệu gốc để kiểm chứng độc lập
    train_df = pd.read_csv(ROOT_DIR / "data" / "raw" / "train.csv", encoding="utf-8-sig", keep_default_na=False)
    test_df = pd.read_csv(ROOT_DIR / "data" / "raw" / "test.csv", encoding="utf-8-sig", keep_default_na=False)
    y_true = train_df["label"].map(LABEL2ID).values

    # 3. So sánh metrics.json giữa V1 và V2
    if v2_metrics_path.exists() and (V1_DIR / "metrics.json").exists():
        v1_m = json.loads((V1_DIR / "metrics.json").read_text(encoding="utf-8"))
        v2_m = json.loads(v2_metrics_path.read_text(encoding="utf-8"))

        v1_base = v1_m.get("oof_macro_f1_base", 0.0)
        v1_tuned = v1_m.get("oof_macro_f1_tuned", 0.0)
        v2_base = v2_m.get("oof_macro_f1_base", 0.0)
        v2_global_tuned = v2_m.get("oof_macro_f1_global_tuned", v2_m.get("oof_macro_f1_tuned", 0.0))
        v2_tuned = v2_m.get("oof_macro_f1_tuned", 0.0)

        print("\n" + "-" * 78)
        print("📊 1. SO SÁNH ĐIỂM 5-FOLD OOF MACRO-F1 (14,000 MẪU)")
        print("-" * 78)
        print(f"  - OOF Base (Argmax)        : V1 = {v1_base:.5f}  |  V2 = {v2_base:.5f}  ({v2_base - v1_base:+.5f})")
        print(f"  - OOF Global Multipliers   : V1 = {v1_tuned:.5f}  |  V2 = {v2_global_tuned:.5f}  ({v2_global_tuned - v1_tuned:+.5f})")
        print(f"  - OOF Final Tuned (PerLang): V1 = {v1_tuned:.5f}  |  V2 = {v2_tuned:.5f}  ({v2_tuned - v1_tuned:+.5f})")

        print("\n📊 2. SO SÁNH ĐIỂM THEO TỪNG FOLD & CONVERGENCE")
        print("-" * 78)
        for k in range(5):
            fk = f"fold_{k}"
            f1_v1 = v1_m.get("per_fold", {}).get(fk, {}).get("best_f1", 0.0)
            f1_v2 = v2_m.get("per_fold", {}).get(fk, {}).get("best_f1", 0.0)
            ep_v1 = v1_m.get("per_fold", {}).get(fk, {}).get("best_epoch_step", "N/A")
            ep_v2 = v2_m.get("per_fold", {}).get(fk, {}).get("best_epoch_step", "N/A")
            bad_v2 = v2_m.get("per_fold", {}).get(fk, {}).get("n_bad_steps", 0)
            print(
                f"  - Fold {k}: V1 = {f1_v1:.5f} ({ep_v1}) | "
                f"V2 = {f1_v2:.5f} ({ep_v2}) [{f1_v2 - f1_v1:+.5f}] (NaN steps: {bad_v2})"
            )

        print("\n📊 3. SO SÁNH THEO NGÔN NGỮ (ĐẶC BIỆT TIẾNG BỒ ĐÀO NHA 'pt' CHIẾM 73% LỖI V1)")
        print("-" * 78)
        for lang in ["pt", "fr", "it", "en"]:
            l_v1 = v1_m.get("per_language_f1", {}).get(lang, 0.0)
            l_v2 = v2_m.get("per_language_f1", {}).get(lang, 0.0)
            m_v2 = v2_m.get("lang_multipliers", {}).get(lang, v2_m.get("multipliers", []))
            m_str = str([round(x, 3) for x in m_v2]) if m_v2 else "N/A"
            print(f"  - Lang '{lang}': V1 = {l_v1:.5f} -> V2 = {l_v2:.5f} ({l_v2 - l_v1:+.5f}) | Multipliers V2: {m_str}")

        print("\n📊 4. SO SÁNH THEO TỪNG CLASS (BENIGN / JAILBREAK / PROMPT_INJECTION)")
        print("-" * 78)
        for cls_name in VALID_LABELS:
            c_v1 = v1_m.get("per_class_f1_tuned", {}).get(cls_name, 0.0)
            c_v2 = v2_m.get("per_class_f1_tuned", {}).get(cls_name, 0.0)
            print(f"  - {cls_name:<18}: V1 = {c_v1:.5f} -> V2 = {c_v2:.5f} ({c_v2 - c_v1:+.5f})")

    # 4. Phân tích ma trận lỗi trên OOF nếu có oof_probs_rembert.npy
    if v2_oof_path.exists() and (V1_DIR / "oof_probs_rembert.npy").exists():
        v1_oof = np.load(V1_DIR / "oof_probs_rembert.npy")
        v2_oof = np.load(v2_oof_path)
        assert v2_oof.shape == (14000, 3), f"Sai kích thước OOF V2: {v2_oof.shape}"
        assert not np.isnan(v2_oof).any(), "OOF V2 chứa NaN!"

        v1_err = int((v1_oof.argmax(axis=1) != y_true).sum())
        v2_err = int((v2_oof.argmax(axis=1) != y_true).sum())
        pt_mask = (train_df["language"] == "pt").values
        v1_pt_err = int((v1_oof[pt_mask].argmax(axis=1) != y_true[pt_mask]).sum())
        v2_pt_err = int((v2_oof[pt_mask].argmax(axis=1) != y_true[pt_mask]).sum())

        print("\n" + "-" * 78)
        print("🔍 5. PHÂN TÍCH SỐ CA ĐOÁN SAI TRÊN OOF (ERROR COUNT ANALYSIS)")
        print("-" * 78)
        print(f"  - Tổng số ca sai toàn tập (Argmax): V1 = {v1_err} ca | V2 = {v2_err} ca ({v2_err - v1_err:+d} ca)")
        print(f"  - Số ca sai riêng tiếng Bồ Đào Nha: V1 = {v1_pt_err} ca | V2 = {v2_pt_err} ca ({v2_pt_err - v1_pt_err:+d} ca)")

    # 5. Kiểm định chất lượng file submission.csv của V2 và đối chiếu với V1
    if v2_sub_path.exists():
        sub_v2 = pd.read_csv(v2_sub_path, dtype={"id": str})
        print("\n" + "-" * 78)
        print("🛡️ 6. KIỂM ĐỊNH SANITY CHECK & ĐỐI CHIẾU DỰ ĐOÁN TEST (6,000 MẪU)")
        print("-" * 78)
        assert len(sub_v2) == 6000, f"Số dòng phải là 6000, thực tế: {len(sub_v2)}"
        assert list(sub_v2.columns) == ["id", "label"], f"Sai tên cột: {list(sub_v2.columns)}"
        assert not sub_v2.isna().any().any(), "Có giá trị NaN trong submission.csv!"
        assert set(sub_v2["label"].unique()).issubset(set(VALID_LABELS)), f"Nhãn lạ: {set(sub_v2['label'].unique())}"
        assert (sub_v2["id"].values == test_df["id"].astype(str).values).all(), "Thứ tự ID không khớp test.csv!"

        dist_v2 = sub_v2["label"].value_counts()
        pct_v2 = sub_v2["label"].value_counts(normalize=True) * 100
        for lbl in VALID_LABELS:
            print(f"  - V2 Test {lbl:<18}: {dist_v2.get(lbl, 0):>5,} dòng ({pct_v2.get(lbl, 0.0):>5.2f}%)")

        v1_sub_path = V1_DIR / "submission.csv"
        if v1_sub_path.exists():
            sub_v1 = pd.read_csv(v1_sub_path, dtype={"id": str})
            diff_mask = sub_v1["label"] != sub_v2["label"]
            n_diff = int(diff_mask.sum())
            agreement = (1.0 - n_diff / 6000.0) * 100.0
            print(f"\n  - Độ đồng thuận giữa V2 và V1 (LB 0.97058): {agreement:.2f}% (Đổi nhãn {n_diff} / 6,000 mẫu)")
            if n_diff > 0:
                df_comp = pd.DataFrame({
                    "lang": test_df["language"],
                    "v1": sub_v1["label"],
                    "v2": sub_v2["label"],
                })
                diff_by_lang = df_comp[diff_mask]["lang"].value_counts().to_dict()
                print(f"  - Phân bổ {n_diff} mẫu đổi nhãn theo ngôn ngữ: {diff_by_lang}")
                ct = pd.crosstab(df_comp[diff_mask]["v1"], df_comp[diff_mask]["v2"])
                print("\n  - Ma trận chuyển dịch nhãn (Dòng = V1 cũ, Cột = V2 mới):")
                print(ct.to_string())

    print("\n" + "=" * 78)
    print("✅ KẾT LUẬN & KHUYẾN NGHỊ HÀNH ĐỘNG CHO GATEKEEPER (NHẬT TÂN):")
    print("  1. Nếu V2 OOF Macro-F1 >= 0.9655 và số mẫu đổi nhãn so với V1 nằm trong khoảng 20 - 180 mẫu:")
    print("     -> ĐỦ ĐIỀU KIỆN NỘP NGAY LƯỢT 4/6 TỪ KAGGLE NOTEBOOK OUTPUT ĐỂ ĐÁNH CHIẾM TOP 1!")
    print("  2. Nếu chỉ 1-2 fold cải thiện vượt trội còn fold khác bị nhiễu ở epoch 4:")
    print("     -> Xem xét K-fold weight/selection của CÙNG backbone google/rembert trước khi nộp.")
    print("=" * 78)


if __name__ == "__main__":
    main()
