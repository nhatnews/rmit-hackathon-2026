# 🎯 BẢNG PHÂN CÔNG NHIỆM VỤ & CHIẾN THUẬT — RMIT HACKATHON 2026 (CHALLENGE 1)

> [!IMPORTANT]
> **Đọc ngay khi vừa `git pull` repo về máy:**
>
> Toàn bộ dữ liệu thật (`data/raw/`), bộ chia 5-Fold cố định (`data/processed/folds.csv`), và mô hình Baseline 3-Class đầu tiên (**OOF Macro-F1 = `0.9332`**) **đã có sẵn trong repo**. Không ai cần tải lại dữ liệu hay tự chia lại K-Fold!

---

## 📊 1. Hiện Trạng Repo & Điểm Benchmark Hiện Tại

| Thành phần | Trạng thái | Đường dẫn file trong Repo | Ghi chú quan trọng |
| :--- | :---: | :--- | :--- |
| **Dữ liệu gốc (`train`, `test`)** | ✅ Sẵn sàng | `data/raw/train.csv` (14,000 dòng)<br>`data/raw/test.csv` (6,000 dòng) | Chỉ đọc (`READ-ONLY`), không có giá trị `NaN`. |
| **Bộ chia 5-Fold dùng chung** | ✅ Đã khóa | `data/processed/folds.csv` | **Bắt buộc cả 4 thành viên dùng chung** qua `load_folds(train_df)`. |
| **Baseline #1 (TF-IDF + LogReg)** | ✅ Đã chạy | `submissions/baseline_tfidf_logreg.csv`<br>`data/processed/oof_tfidf_logreg.npy` | **OOF Macro-F1 = `0.9250`** (sau khi tối ưu hệ số nhân 3 lớp). |
| **Baseline #2 (Blend LogReg + LinearSVC + Per-Lang Multipliers)** | ✅ Đã chạy | `submissions/baseline_tfidf_blend.csv`<br>`data/processed/oof_tfidf_svc.npy` | **OOF Macro-F1 = `0.9332`** (`fr`: 0.9539, `it`: 0.9382, `en`: 0.9352, `pt`: 0.9179). |

---

## 💎 2. Phát Hiện Sống Còn Từ Error Analysis Trên 14,000 Mẫu Thật

Sau khi kiểm tra Confusion Matrix trên toàn bộ 14,000 mẫu của tập `train.csv`, đây là **bức tranh thật sự của dữ liệu** mà cả 4 thành viên phải bám sát:

| Nhãn Thật \ Dự Đoán (Baseline) | `BENIGN` | `JAILBREAK` | `PROMPT_INJECTION` | Điểm nghẽn cần giải quyết |
| :--- | :---: | :---: | :---: | :--- |
| **`BENIGN` (8,680)** | **8,298** | **339** ❌ | 43 ❌ | Nhầm sang `JAILBREAK` tập trung ở tiếng Bồ Đào Nha (`pt`). |
| **`JAILBREAK` (4,340)** | **304** ❌ | **4,030** | 6 | Nhầm về `BENIGN` tập trung ở tiếng Bồ Đào Nha (`pt`). |
| **`PROMPT_INJECTION` (980)** | **101** ❌ | 10 | **869** | Nhầm về `BENIGN` ở `en, it, fr`. Hầu như **không nhầm** với `JAILBREAK`! |

### 3 Quy Luật "Mỏ Vàng" Quyết Định Thứ Hạng:
1. **Nhầm lẫn giữa `JAILBREAK` và `PROMPT_INJECTION` gần như bằng 0 (chỉ 16/14,000 mẫu):**
   - Không cần tốn thời gian viết phức tạp để tách `JAILBREAK` khỏi `PROMPT_INJECTION`.
2. **80% tổng số lỗi của toàn bộ dữ liệu (643/803 lỗi) nằm ở `BENIGN` $\leftrightarrow$ `JAILBREAK` trong tiếng Bồ Đào Nha (`pt` — chiếm 45% dataset):**
   - Ở `fr, it, en`, F1 của `JAILBREAK` đã đạt **0.980 – 0.986**, nhưng ở `pt` chỉ đạt **0.8547**!
   - **Lý do:** Tập `pt` có rất nhiều câu **Contrastive Benign** (mượn văn phong nhập vai kiểu *"Imagine que você é um assistente de IA travesso..."* nhưng lại hỏi việc đời thường vô hại) VÀ các câu **Subtle Jailbreak** (đóng vai nhà sử học/học thuật lịch sự *"Suponha que você seja um historiador..."* nhưng hỏi nội dung bị cấm).
3. **Quy luật Độ Dài Vàng (`char_len`):**
   - **`JAILBREAK`:** 100% mẫu nằm gọn trong đoạn **`[112, 3752]` ký tự** (0% mẫu `JAILBREAK` nào ngắn hơn 112 ký tự hoặc dài hơn 3,752 ký tự!).
   - **`BENIGN`:** 100% các mẫu dài trên **`5,128` ký tự** (tối đa 18,550 ký tự) đều là `BENIGN`!
   - **`PROMPT_INJECTION`:** Ngắn gọn (trung vị chỉ **`99` ký tự**, 75% dưới `263` ký tự).

---

## 📋 3. Danh Sách 10 Approved Models Chính Thức Của BTC

> [!WARNING]
> **Quy định bắt buộc:** Chỉ được phép dùng các pretrained models trong bảng dưới đây. **Bắt buộc phải ghi rõ tên model được chọn ngay ở đầu Kaggle Notebook** và đặt tên notebook theo chuẩn `<university>_<campus>_<team>_C1` (VD: `RMIT_SGS_<team>_C1`).

| # | Tên Mô Hình (`model_name`) | Loại | Link Kaggle Model | Phân công trong Team |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `protectai/deberta-v3-base-prompt-injection-v2` | Encoder (Safety) | [Kaggle Link](https://www.kaggle.com/models/aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default) | **Trung Quân** (Ưu tiên #1) |
| 2 | `FacebookAI/xlm-roberta-large` | Encoder (Best) | [Kaggle Link](https://www.kaggle.com/models/aisuko/xlm-roberta-large/Transformers/default) | **Trung Quân** (Ưu tiên #2) |
| 3 | `FacebookAI/xlm-roberta-base` | Encoder | [Kaggle Link](https://www.kaggle.com/models/aisuko/xlm-roberta-base/Transformers/default) | **Trung Quân** (Ưu tiên #2) |
| 4 | `patronus-studio/wolf-defender-prompt-injection` | Encoder (Safety) | [Kaggle Link](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default) | **Trung Quân** (Ưu tiên #3) |
| 5 | `jhu-clsp/mmBERT-base` | Encoder (Fast) | [Kaggle Link](https://www.kaggle.com/models/aisuko/mmbert-base/Transformers/default) | **Nhật Tân** (Chia tải GPU) |
| 6 | `jhu-clsp/mmBERT-small` | Encoder (Fast) | [Kaggle Link](https://www.kaggle.com/models/aisuko/mmbert-small/Transformers/default) | **Nhật Tân** (Chia tải GPU) |
| 7 | `distilbert/distilbert-base-multilingual-cased` | Encoder (Fastest) | [Kaggle Link](https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/Transformers/default) | **Nhật Tân** (Chia tải GPU) |
| 8 | `google/rembert` | Encoder (Batch=8) | [Kaggle Link](https://www.kaggle.com/models/aisuko/rembert/Transformers/default) | Dự phòng đa dạng hóa |
| 9 | `ibm-granite/granite-3.3-2b-instruct` | LLM (LoRA) | [Kaggle Link](https://www.kaggle.com/models/aisuko/granite-3.3-2b-instruct/Transformers/default) | Dự phòng LLM LoRA |
| 10 | `mistralai/Mistral-7B-Instruct-v0.3` | LLM (LoRA) | [Kaggle Link](https://www.kaggle.com/models/aisuko/mistral-7b-instruct-v0.3/Transformers/default) | Dự phòng LLM LoRA |

---

## 👥 4. Việc Cần Làm Ngay Theo Từng Thành Viên

> 💡 **Mẹo dùng AI trợ lý (Antigravity / Cursor / Claude / Copilot):**
>
> Mở file [`prompt.md`](file:///d:/Code/rmit-hackathon-2026/prompt.md), copy phần Prompt cá nhân đúng tên bạn gửi cho AI để AI nắm rõ 100% quy tắc và viết code đúng chuẩn repo.

---

### 👑 1. Nhật Tân — Team Lead, Fast Encoders & Ensemble Master
**File làm việc:** `notebooks/01_tan_ensemble_error_analysis.ipynb` & `src/blend.py`

- [ ] **Task 1 (Quản lý Quota 6 lần nộp):** Dùng **Lượt nộp #1/6** với file [`submissions/baseline_tfidf_blend.csv`](file:///d:/Code/rmit-hackathon-2026/submissions/baseline_tfidf_blend.csv) (OOF Macro-F1 = `0.9332`) để xác nhận pipeline và lấy điểm mốc trên Public Leaderboard. **Giữ chặt 5 lượt nộp còn lại!**
- [ ] **Task 2 (Chia tải GPU — Chạy nhánh Fast Multilingual Encoders):**
  - Mở Kaggle GPU cá nhân, fine-tune 5-Fold cho **`jhu-clsp/mmBERT-base`** (hoặc `distilbert/distilbert-base-multilingual-cased`).
  - Xuất `oof_mmbert.npy` `(14000, 3)` và `test_mmbert.npy` `(6000, 3)` bỏ vào `data/processed/`.
- [ ] **Task 3 (Ensembling & Tối ưu Ngưỡng Theo Ngôn Ngữ `optimize_per_language_multipliers`):**
  - Thu thập toàn bộ các file `oof_*.npy` `(14000, 3)` và `test_*.npy` `(6000, 3)` từ Trung Quân (DeBERTa, XLM-R) và Khánh Linh (LightGBM, CatBoost, Per-Language Models).
  - Chạy `search_weights()` + **`optimize_per_language_multipliers()`** trong `src/blend.py` để tối ưu bộ nhân xác suất riêng cho từng ngôn ngữ (`pt, fr, it, en`).
  - Kiểm định bằng `validate_submission()` trước mỗi lần nộp.

---

### 🔥 2. Trung Quân — Deep Learning & Heavy Approved HF Specialist (Kaggle GPU)
**Môi trường làm việc:** **Kaggle Notebook GPU (T4 x2 hoặc P100)** — *Ưu tiên chạy ngay!*

- [ ] **Task 1 (Khởi tạo Kaggle Notebook chuẩn quy chế):**
  - Đặt tên Notebook đúng quy định: `<university>_<campus>_<team>_C1` (VD: `RMIT_SGS_<team>_C1`).
  - **Ghi rõ tên Approved Model sử dụng ở cell Markdown đầu tiên của Notebook.**
  - Dùng đúng file `data/processed/folds.csv` của nhóm.
- [ ] **Task 2 (Fine-tune các Approved Models chủ lực):**
  1. **Model 1:** `protectai/deberta-v3-base-prompt-injection-v2`
  2. **Model 2:** `FacebookAI/xlm-roberta-base` (hoặc `FacebookAI/xlm-roberta-large`)
  3. **Model 3 (nếu kịp):** `patronus-studio/wolf-defender-prompt-injection`
- [ ] **Task 3 (Kỹ thuật bắt buộc khi Fine-tune):**
  - **Ghép tiền tố ngôn ngữ:** `"lang: " + language + " | " + text` (giúp mô hình nhận diện ngay ngữ cảnh tiếng Bồ Đào Nha `pt` để xử lý *Contrastive Benign vs Subtle Jailbreak*).
  - **Head + Tail Truncation (`max_length=512`):** Lấy **256 token đầu + 256 token cuối** cho văn bản dài (tới 18,550 ký tự).
  - **Xử lý mất cân bằng 62% - 31% - 7%:** Dùng `CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 6.0]).cuda())` hoặc Focal Loss.
- [ ] **Task 4 (Xuất file bàn giao cho Nhật Tân):**
  - Lưu xác suất sau `softmax(dim=-1)` theo đúng thứ tự `0: BENIGN, 1: JAILBREAK, 2: PROMPT_INJECTION`:
    - `oof_deberta.npy` `(14000, 3)` & `test_deberta.npy` `(6000, 3)`
    - `oof_xlmr.npy` `(14000, 3)` & `test_xlmr.npy` `(6000, 3)`

---

### 🛡️ 3. Cường Thịnh — Multilingual Security & NLP Feature Engineering Specialist
**File làm việc:** `notebooks/01_thinh_features.ipynb` & [`src/data_loader.py`](file:///d:/Code/rmit-hackathon-2026/src/data_loader.py)

- [ ] **Task 1 (Giải quyết Điểm Nghẽn #1 — 80% Lỗi nằm ở `BENIGN` vs `JAILBREAK` tiếng Bồ Đào Nha `pt`):**
  - Xây dựng nhóm đặc trưng phân biệt **Contrastive Benign** (mượn từ khóa nhập vai *"Imagine que você é..."*, *"sem diretrizes éticas"* nhưng yêu cầu bên dưới hoàn toàn vô hại) với **Subtle Jailbreak** (đóng vai nhà sử học/nhà nghiên cứu *"Suponha que você seja um historiador..."*, *"universo alternativo"* nhưng chứa từ khóa độc hại: gian lận, mã độc, bạo lực, phân biệt đối xử, thuốc/độc chất...).
- [ ] **Task 2 (Giải quyết Điểm Nghẽn #2 — 18% Lỗi nằm ở `BENIGN` vs `PROMPT_INJECTION` trên `en, it, fr`):**
  - Bổ sung regex bắt các câu lệnh tiêm nhiễm ngắn (`char_len < 260`), chuyển hướng tác vụ đột ngột, lệnh yêu cầu tiết lộ/bỏ qua chỉ thị trước đó bằng tiếng Pháp, Ý, Anh.
- [ ] **Task 3 (Khai thác Ngưỡng Độ Dài Vàng & Thống kê Cấu trúc):**
  - Đã có sẵn trong `src/data_loader.py`: `is_len_lt_112`, `is_len_112_to_3752`, `is_len_gt_3752`, `is_len_gt_5128`.
  - Bổ sung thêm: tỷ lệ độ dài câu cuối so với toàn bài, số lượng câu hỏi (`?`), mật độ từ khóa độc hại ở nửa sau của prompt (nơi chứa yêu cầu thật sự sau phần dẫn chuyện roleplay!).
- [ ] **Task 4 (Xuất Feature Matrix bàn giao cho Khánh Linh):**
  - Lưu thành `data/processed/train_features.parquet` (hoặc `.csv`) & `data/processed/test_features.parquet`.

---

### ⚡ 4. Khánh Linh — Tabular ML, Per-Language Models & Stacking Specialist
**File làm việc:** `notebooks/02_linh_tabular.ipynb`

- [ ] **Task 1 (Xây dựng Mô hình Chuyên biệt cho Tiếng Bồ Đào Nha `pt`):**
  - Vì `pt` chiếm tới **45% dữ liệu** (`6,300` train / `2,700` test) và chứa **80% tổng số lỗi**, hãy train riêng một pipeline `TF-IDF (max_features=40000) + LinearSVC / LightGBM` **chỉ trên các dòng `language == "pt"`** (và tương tự cho `fr, it, en`) để từ vựng tiếng Bồ Đào Nha không bị pha loãng bởi 3 ngôn ngữ kia!
- [ ] **Task 2 (Huấn luyện LightGBM / CatBoost Stacking 3-Class trên 5-Fold `folds.csv`):**
  - Kết hợp:
    1. Đặc trưng bảo mật & ngưỡng độ dài vàng từ `extract_adversarial_features()` của Cường Thịnh.
    2. `TruncatedSVD(n_components=64)` từ ma trận TF-IDF.
    3. Xác suất Out-of-Fold từ các mô hình tuyến tính (`oof_tfidf_logreg.npy`, `oof_tfidf_svc.npy`) làm meta-features.
  - Huấn luyện `LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced", learning_rate=0.03, num_leaves=31)`.
- [ ] **Task 3 (Xuất file bàn giao cho Nhật Tân):**
  - Lưu `data/processed/oof_lgbm.npy` `(14000, 3)` và `data/processed/test_lgbm.npy` `(6000, 3)`.
  - Xuất file dự đoán vào `submissions/lgbm_multilingual.csv` và kiểm định bằng `validate_submission()`.

---

## 🤝 5. Giao Thức Phối Hợp Chung (Standard Handshake Protocol)

1. **Thứ tự dòng:** Giữ nguyên 100% thứ tự dòng gốc của `train.csv` (`14,000` dòng) và `test.csv` (`6,000` dòng).
2. **Bộ chia Fold:** Luôn đọc từ `folds = load_folds(train_df)`.
3. **Thứ tự 3 cột trong ma trận xác suất `(N, 3)`:**
   - `proba[:, 0]` $\rightarrow$ **`BENIGN`**
   - `proba[:, 1]` $\rightarrow$ **`JAILBREAK`**
   - `proba[:, 2]` $\rightarrow$ **`PROMPT_INJECTION`**
4. **Kiểm định trước khi nộp (Bảo vệ 6 lượt nộp):**
   ```python
   from src.utils import save_submission
   save_submission(sub_df, filename="ten_file.csv", test_df=test_df)
   ```
