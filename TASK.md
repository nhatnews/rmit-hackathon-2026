# 🎯 BẢNG PHÂN CÔNG NHIỆM VỤ & CHIẾN THUẬT — RMIT HACKATHON 2026 (CHALLENGE 1)

> [!IMPORTANT]
> **Đọc ngay khi vừa `git pull` repo về máy:**
> Toàn bộ dữ liệu thật (`data/raw/`), bộ chia 5-Fold cố định (`data/processed/folds.csv`), và mô hình Baseline 3-Class đầu tiên (**OOF Macro-F1 = `0.9316`**) **đã có sẵn trong repo**. Không ai cần tải lại dữ liệu hay tự chia lại K-Fold!

---

## 📊 1. Hiện Trạng Repo & Điểm Benchmark Hiện Tại

| Thành phần | Trạng thái | Đường dẫn file trong Repo | Ghi chú quan trọng |
| :--- | :---: | :--- | :--- |
| **Dữ liệu gốc (`train`, `test`)** | ✅ Sẵn sàng | `data/raw/train.csv` (14,000 dòng)<br>`data/raw/test.csv` (6,000 dòng) | Chỉ đọc (`READ-ONLY`), không có giá trị `NaN`. |
| **Bộ chia 5-Fold dùng chung** | ✅ Đã khóa | `data/processed/folds.csv` | **Bắt buộc cả 4 thành viên dùng chung** qua `load_folds(train_df)`. |
| **Baseline #1 (TF-IDF + LogReg)** | ✅ Đã chạy | `submissions/baseline_tfidf_logreg.csv`<br>`data/processed/oof_tfidf_logreg.npy` | **OOF Macro-F1 = `0.9250`** (sau khi tối ưu hệ số nhân 3 lớp). |
| **Baseline #2 (Blend LogReg + LinearSVC)** | ✅ Đã chạy | `submissions/baseline_tfidf_blend.csv`<br>`data/processed/oof_tfidf_svc.npy` | **OOF Macro-F1 = `0.9316`** (`BENIGN`: 0.9547, `JAILBREAK`: 0.9244, `INJECTION`: 0.9157). |

---

## 💎 2. Phát Hiện "Mỏ Vàng" Từ Dữ Liệu Thật (14,000 Mẫu) — Cả Team Cần Nắm

1. **Phân bố Ngôn ngữ (`language`) & Nhãn (`label`) khớp 100% giữa Train và Test:**
   - **Ngôn ngữ:** Tiếng Bồ Đào Nha `pt` (**45%**), Tiếng Pháp `fr` (**25%**), Tiếng Ý `it` (**20%**), Tiếng Anh `en` (**10%**).
   - **Tỷ lệ 3 lớp (giống hệt nhau ở cả 4 ngôn ngữ):**
     - `BENIGN`: **62.0%** (`8,680` mẫu train)
     - `JAILBREAK`: **31.0%** (`4,340` mẫu train)
     - `PROMPT_INJECTION`: **7.0%** (`980` mẫu train — **chiếm 1/3 tổng điểm Macro-F1!**)
2. **Quy luật Độ dài Văn bản (`char_len`) cực kỳ đắt giá:**
   - **`JAILBREAK`:** Độ dài nằm gọn trong khoảng **`[112, 3752]` ký tự** (trung vị `565` ký tự). **Không có bất kỳ mẫu `JAILBREAK` nào dưới 112 ký tự hoặc trên 3,752 ký tự!**
   - **`PROMPT_INJECTION`:** Ngắn gọn, trung vị chỉ **`99` ký tự** (75% mẫu dưới `263` ký tự, tối đa `5,128` ký tự).
   - **`BENIGN`:** Trải dài từ `10` đến **`18,550` ký tự** (100% mẫu dài trên `5,128` ký tự trong tập train là `BENIGN`).

---

## 👥 3. Việc Cần Làm Ngay Theo Từng Thành Viên

> 💡 **Mẹo dùng AI trợ lý (Antigravity / Cursor / Claude / Copilot):**
> Mở file [`prompt.md`](file:///d:/Code/rmit-hackathon-2026/prompt.md), copy phần Prompt cá nhân đúng tên bạn gửi cho AI để AI nắm rõ 100% quy tắc và viết code đúng chuẩn repo.

---

### 👑 1. Nhật Tân — Team Lead, Kaggle Pipeline & Ensemble Master
**File làm việc:** `notebooks/01_tan_ensemble_error_analysis.ipynb` & `src/blend.py`

- [ ] **Task 1 (Quản lý Quota 6 lần nộp):** Kiểm tra kỹ và dùng **Lượt nộp #1/6** trên Kaggle bằng file [`submissions/baseline_tfidf_blend.csv`](file:///d:/Code/rmit-hackathon-2026/submissions/baseline_tfidf_blend.csv) (OOF Macro-F1 = `0.9316`) để xác nhận pipeline và lấy điểm mốc trên Public Leaderboard. **Giữ chặt 5 lượt nộp còn lại!**
- [ ] **Task 2 (Error Analysis trên OOF):**
  - Load `oof_tfidf_svc.npy` và `oof_tfidf_logreg.npy` để in **Confusion Matrix** và xuất ra ~50 mẫu bị dự đoán sai nhiều nhất giữa `JAILBREAK` và `PROMPT_INJECTION`.
  - Gửi danh sách các mẫu sai này cho **Cường Thịnh** để bổ sung từ khóa Regex.
- [ ] **Task 3 (Nhánh Model bổ trợ — Ridge / SGD / Multilingual Char+Word N-gram sâu hơn):**
  - Thử nghiệm `SGDClassifier(loss="modified_huber")` / `ComplementNB` / `MLP` trên đặc trưng TF-IDF để lấy thêm OOF đa dạng.
- [ ] **Task 4 (Ensembling & Tối ưu Ngưỡng Macro-F1 cuối cùng):**
  - Thu thập toàn bộ các file `oof_*.npy` `(14000, 3)` và `test_*.npy` `(6000, 3)` từ Trung Quân (DeBERTa, XLM-R) và Khánh Linh (LightGBM, CatBoost).
  - Chạy `search_weights()` + `optimize_class_multipliers()` trong `src/blend.py` để tìm bộ trọng số model và hệ số nhân lớp $(m_0, m_1, m_2)$ đạt **OOF Macro-F1 cao nhất** (mục tiêu: **> 0.96+ Macro-F1**).
  - Kiểm định bằng `validate_submission()` trước mỗi lần nộp.

---

### 🔥 2. Trung Quân — Deep Learning & Approved Hugging Face Specialist (Kaggle GPU)
**Môi trường làm việc:** **Kaggle Notebook GPU (T4 x2 hoặc P100)** — *Ưu tiên chạy ngay vì tốn thời gian GPU nhất!*

- [ ] **Task 1 (Khởi tạo Kaggle Notebook chuẩn tên BTC):**
  - Tạo Notebook trên Kaggle đặt tên đúng quy định: `<university>_<campus>_<team>_C1` (ví dụ: `RMIT_SGS_<team>_C1`).
  - Upload file `data/processed/folds.csv` lên Kaggle Dataset để dùng **đúng 100% bộ chia 5-Fold của nhóm**.
- [ ] **Task 2 (Fine-tune các Approved Models bắt buộc của BTC):**
  Huấn luyện 5-Fold (hoặc tối thiểu 3–5 Fold) cho 2–3 mô hình mạnh nhất trong danh sách được duyệt:
  1. **Ưu tiên số 1:** `protectai/deberta-v3-base-prompt-injection-v2` (đã pretrain sẵn cho bài toán Prompt Injection!).
  2. **Ưu tiên số 2:** `FacebookAI/xlm-roberta-base` (mạnh cho đa ngôn ngữ `pt, fr, it, en`).
  3. **Ưu tiên số 3 (nếu còn thời gian GPU):** `patronus-studio/wolf-defender-prompt-injection` hoặc `distilbert-base-multilingual-cased`.
- [ ] **Task 3 (Kỹ thuật bắt buộc khi Fine-tune để điểm cao):**
  - **Ghép tiền tố ngôn ngữ:** Đưa đầu vào dạng `"lang: " + language + " | " + text`.
  - **Head + Tail Truncation (`max_length=512`):** Với các văn bản dài (tới 18,550 ký tự), lấy **256 token đầu + 256 token cuối** (không cắt cụt phần đuôi vì lệnh injection thường nằm ở cuối!).
  - **Xử lý mất cân bằng 62% - 31% - 7%:** Sử dụng `CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 6.0]).cuda())` hoặc Focal Loss.
- [ ] **Task 4 (Xuất file bàn giao cho Nhật Tân):**
  - Lưu xác suất sau hàm `softmax(dim=-1)` theo đúng thứ tự nhãn `0: BENIGN, 1: JAILBREAK, 2: PROMPT_INJECTION`:
    - `oof_deberta.npy` kích thước `(14000, 3)` và `test_deberta.npy` kích thước `(6000, 3)`
    - `oof_xlmr.npy` kích thước `(14000, 3)` và `test_xlmr.npy` kích thước `(6000, 3)`

---

### 🛡️ 3. Cường Thịnh — Multilingual Security & NLP Feature Engineering Specialist
**File làm việc:** `notebooks/01_thinh_features.ipynb` & cập nhật hàm `extract_adversarial_features()` trong [`src/data_loader.py`](file:///d:/Code/rmit-hackathon-2026/src/data_loader.py)

- [ ] **Task 1 (Đặc trưng Ngưỡng Độ Dài Vàng từ EDA):**
  Bổ sung các biến chỉ báo (boolean/numeric) khai thác quy luật độ dài:
  - `is_len_lt_112`: `(char_len < 112)` $\rightarrow$ Tín hiệu loại trừ 100% lớp `JAILBREAK`!
  - `is_len_gt_3752`: `(char_len > 3752)` $\rightarrow$ Tín hiệu loại trừ 100% lớp `JAILBREAK`!
  - `is_len_gt_5128`: `(char_len > 5128)` $\rightarrow$ Tín hiệu chỉ báo 100% lớp `BENIGN`!
  - `char_len_zscore_by_lang`: Độ dài chuẩn hóa theo từng ngôn ngữ (`pt, fr, it, en`).
- [ ] **Task 2 (Từ điển Regex Đa Ngôn Ngữ `pt, fr, it, en`):**
  - Mở rộng các nhóm từ khóa trong [`docs/cheat_sheet_security_low_resource.md`](file:///d:/Code/rmit-hackathon-2026/docs/cheat_sheet_security_low_resource.md) để bắt chính xác:
    - **Từ khóa `PROMPT_INJECTION`:** Lệnh ghi đè (*ignore previous / ignorez les instructions / ignora le istruzioni / ignore as instruções*), đánh cắp system prompt, chuyển đổi tác vụ đột ngột (`new task`, `nouvelle tâche`, `nuovo compito`, `nova tarefa`).
    - **Từ khóa `JAILBREAK`:** Nhập vai persona (`DAN`, `AIM`, `STAN`, `mode développeur`, `modalità sviluppatore`, `modo desenvolvedor`), kịch bản giả định/hư cấu, mã hóa Base64/Hex/Leetspeak.
- [ ] **Task 3 (Đặc trưng Cấu trúc & Vị trí Đầu/Cuối Câu):**
  - Tỷ lệ dấu chấm than, dấu ngoặc kép, dấu hai chấm, mật độ viết hoa, số dòng trống (`\n\n`), sự xuất hiện của từ khóa lệnh ở **300 ký tự đầu** so với **300 ký tự cuối**.
- [ ] **Task 4 (Xuất Feature Matrix bàn giao cho Khánh Linh):**
  - Đảm bảo mọi cột đặc trưng đều là số (`float`/`int`), không có `NaN`/`inf`, và lưu thành `data/processed/train_features.parquet` (hoặc `.csv`) & `data/processed/test_features.parquet`.

---

### ⚡ 4. Khánh Linh — Tabular ML & Multiclass TF-IDF Specialist
**File làm việc:** `notebooks/02_linh_tabular.ipynb`

- [ ] **Task 1 (Xây dựng Feature Matrix Kết hợp Sparse + Dense):**
  - Kết hợp:
    1. Ma trận thưa **TF-IDF đa ngôn ngữ** (Word n-grams `(1, 3)` + Char `char_wb` n-grams `(2, 5)`) + `TruncatedSVD(n_components=64)`.
    2. Xác suất Out-of-Fold từ các mô hình tuyến tính (`oof_tfidf_logreg.npy`, `oof_tfidf_svc.npy`) làm meta-features (Stacking features!).
    3. Toàn bộ **bảng đặc trưng bảo mật & độ dài** từ `extract_adversarial_features()` của Cường Thịnh.
- [ ] **Task 2 (Huấn luyện LightGBM & CatBoost 3-Class trên 5-Fold `folds.csv`):**
  - Thiết lập `LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced", learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, subsample=0.8)`.
  - Đánh giá trên đúng `folds = load_folds(train_df)`, theo dõi **OOF Macro-F1** và F1 của lớp `PROMPT_INJECTION`.
- [ ] **Task 3 (Tối ưu Hyperparameters & Kiểm tra Feature Importance):**
  - In bảng Top 30 features quan trọng nhất của LightGBM (gain importance) để báo lại cho Cường Thịnh biết đặc trưng nào đang hiệu quả nhất.
- [ ] **Task 4 (Xuất file bàn giao cho Nhật Tân):**
  - Lưu `data/processed/oof_lgbm.npy` `(14000, 3)` và `data/processed/test_lgbm.npy` `(6000, 3)`.
  - Xuất file dự đoán vào `submissions/lgbm_multilingual.csv` và chạy `validate_submission()` để kiểm tra.

---

## 🤝 4. Giao Thức Phối Hợp Chung (Standard Handshake Protocol)

Để quá trình ghép mô hình (Blending/Stacking) của cả 4 người diễn ra mượt mà, không bị lệch dòng hay lệch nhãn:

1. **Thứ tự dòng:** Tuyệt đối giữ nguyên thứ tự dòng gốc của `train.csv` (`14,000` dòng) và `test.csv` (`6,000` dòng). Không `sort` hay `drop` dòng khi xuất `oof_*.npy` và `test_*.npy`.
2. **Bộ chia Fold:** Luôn sử dụng:
   ```python
   from src.folds import load_folds
   folds = load_folds(train_df)
   ```
3. **Thứ tự 3 cột trong ma trận xác suất `(N, 3)`:**
   Mọi mảng xác suất `oof_<ten_model>.npy` và `test_<ten_model>.npy` phải có kích thước `(14000, 3)` và `(6000, 3)` theo đúng bảng mã trong `src/config.py`:
   - `proba[:, 0]` $\rightarrow$ Xác suất lớp **`BENIGN`**
   - `proba[:, 1]` $\rightarrow$ Xác suất lớp **`JAILBREAK`**
   - `proba[:, 2]` $\rightarrow$ Xác suất lớp **`PROMPT_INJECTION`**
4. **Kiểm định trước khi nộp (Bảo vệ 6 lượt nộp):**
   ```python
   from src.utils import save_submission
   save_submission(sub_df, filename="ten_file.csv", test_df=test_df)
   ```
