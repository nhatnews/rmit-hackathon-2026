# 🚀 RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification

> **Saigon South Campus — Kaggle Competition**  
> Metric: **Macro-averaged F1** | Target Languages: **pt (45%), fr (25%), it (20%), en (10%)**  
> Submission Limit: **Tối đa 6 lần nộp cho toàn bộ cuộc thi**  
> ⚠️ **QUY TẮC CỐT LÕI (STRICT RULE): CHỈ ĐƯỢC XÂY DỰNG GIẢI PHÁP TRÊN ĐÚNG 1 MODEL DUY NHẤT (NO ENSEMBLING/BLENDING)**

---

> 🚨 **QUY ĐỊNH BẮT BUỘC TỪ BAN TỔ CHỨC:**  
> *"You must build your solution on one of the following approved models. They are all open-source-licence safe and provided to you as a Kaggle dataset — no other pretrained backbone is allowed."*  
> - **Chỉ được dùng duy nhất 1 model:** Mọi submission và notebook nộp bài phải xây dựng trên đúng **1 pretrained backbone** được duyệt.
> - **Tuyệt đối cấm Ensemble đa model:** Không được kết hợp (blend/stack/ensemble) nhiều mô hình khác nhau (ví dụ: cấm blend DeBERTa + XLM-RoBERTa). Điểm số từ giải pháp kết hợp nhiều backbone sẽ phạm quy.
> - **Khai báo model:** Ghi rõ tên model ở dòng đầu tiên của notebook Kaggle (ví dụ: `# Model: protectai/deberta-v3-base-prompt-injection-v2`).

---

## 📌 Bối cảnh đề thi & Dữ liệu

- **Bài toán phân loại 3 lớp (`label`):**
  - `BENIGN` (~62% train): An toàn, yêu cầu bình thường.
  - `JAILBREAK` (~31% train): Đóng vai / giả định / thao túng vượt rào bảo vệ (DAN persona, roleplay).
  - `PROMPT_INJECTION` (~7% train — **lớp quyết định điểm số**): Chiếm quyền điều khiển hệ thống, trích xuất prompt.
- **Metric đánh giá:** **Macro-F1** ($\frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} + F1_{\text{PROMPT\_INJECTION}})$). 
  - *Lớp 7% có trọng số điểm ngang bằng lớp 62%, do đó bắt buộc phải dùng Class Weights hoặc tối ưu ngưỡng xác suất.*
- **Dữ liệu:**
  - `train.csv` (14.000 dòng): `id`, `text`, `language`, `label`
  - `test.csv` (6.000 dòng): `id`, `text`, `language`
  - `sample_submission.csv` (6.000 dòng): `id`, `label`

---

## 🛡️ Danh sách 10 Approved Models của BTC (Whitelist) & Ràng Buộc "Chỉ 1 Model"

Tất cả 10 mô hình đều an toàn về bản quyền mã nguồn mở và được cung cấp sẵn dưới dạng Kaggle dataset. **Không được sử dụng bất kỳ pretrained backbone nào khác ngoài danh sách.**

### 1. Nhóm Chuyên An Toàn Bảo Mật (Ưu tiên số 1 cho Prompt Injection)
- [`protectai/deberta-v3-base-prompt-injection-v2`](https://www.kaggle.com/models/aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default) — Đã pre-train chuyên biệt cho Prompt Injection & Jailbreak.
- [`patronus-studio/wolf-defender-prompt-injection`](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default) — Mô hình phân loại an toàn chuyên dụng.

### 2. Nhóm Đa Ngôn Ngữ Hạng Nặng (Ưu tiên giải quyết tiếng Bồ Đào Nha `pt`, Pháp `fr`, Ý `it`)
- [`FacebookAI/xlm-roberta-base`](https://www.kaggle.com/models/aisuko/xlm-roberta-base/Transformers/default) — Cực kỳ ổn định, train nhanh (~35 phút), rất mạnh về Romance languages.
- [`FacebookAI/xlm-roberta-large`](https://www.kaggle.com/models/aisuko/xlm-roberta-large/Transformers/default) — 560M params, sâu nhất, mạnh nhất; cần batch size 4–8 + FP16.
- [`google/rembert`](https://www.kaggle.com/models/aisuko/rembert/Transformers/default) — 32 layers, chuyên truyền tri thức đa ngôn ngữ; batch size 8.

### 3. Nhóm Siêu Tốc (Backup khẩn cấp / Iteration nhanh)
- [`jhu-clsp/mmBERT-base`](https://www.kaggle.com/models/aisuko/mmbert-base/Transformers/default) — Kiến trúc hiện đại, đa ngôn ngữ tốt, train ~20 phút.
- [`jhu-clsp/mmBERT-small`](https://www.kaggle.com/models/aisuko/mmbert-small/Transformers/default) — Rất nhẹ và nhanh.
- [`distilbert/distilbert-base-multilingual-cased`](https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/Transformers/default) — Nhanh nhất (~12 phút).

### 4. Nhóm LLM Fine-Tuning / LoRA
- [`ibm-granite/granite-3.3-2b-instruct`](https://www.kaggle.com/models/aisuko/granite-3.3-2b-instruct/Transformers/default) — LLM 2B instruct.
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://www.kaggle.com/models/aisuko/mistral-7b-instruct-v0.3/Transformers/default) — LLM 7B instruct.

---

## 🎯 Chiến Lược Tác Chiến: "Khám Phá Song Song — Chọn 1 Model Tối Thượng"

1. **Thử nghiệm song song (Parallel Single-Model Exploration):**
   - Các thành viên có thể độc lập chọn 1 mô hình trong danh sách Whitelist để huấn luyện trên Kaggle GPU.
   - Sử dụng chung file mẫu chuẩn hóa: [`notebooks/template_train_approved_model.ipynb`](file:///d:/Code/rmit-hackathon-2026/notebooks/template_train_approved_model.ipynb), chỉ cần thay đổi biến `MODEL_NAME`.
2. **Chọn ra 1 MODEL TỐT NHẤT (Single Champion Model):**
   - Mọi mô hình đều đo điểm trên bộ 5-Fold chung tại [`data/processed/folds.csv`](file:///d:/Code/rmit-hackathon-2026/data/processed/folds.csv) theo **Macro-F1**.
   - Cả đội chọn mô hình có Macro-F1 cao nhất và tối ưu hóa sâu mô hình đó (hyperparameters, head+tail truncation, class weights, post-processing multipliers).
   - **Tuyệt đối không ensemble các mô hình khác nhau.**
3. **Kỷ luật nộp bài (Quota Gatekeeper):**
   - Cả giải **chỉ có 6 lượt nộp**.
   - Mọi lượt nộp do **Nhật Tân** kiểm tra và duyệt bằng `validate_submission()`.
   - Phân bổ: Tối đa 3–4 lượt trong buổi sáng (Session 1), giữ lại ít nhất 2 lượt cho buổi chiều (Session 2 & Final).

---

## ⚡ Hướng Dẫn Chạy Nhanh (Quick Start)

### 1. Chạy Baseline Tabular ML trên máy tính (2 phút)
```bash
# Kích hoạt môi trường
.\.venv\Scripts\activate

# Chạy baseline (OOF Macro-F1 = 0.9332)
python -m src.models
```

### 2. Chạy Transformer trên Kaggle GPU (T4 x2)
1. Mở file [`notebooks/template_train_approved_model.ipynb`](file:///d:/Code/rmit-hackathon-2026/notebooks/template_train_approved_model.ipynb).
2. Tải notebook lên Kaggle, bật **GPU T4 x2**, chọn **Internet On**.
3. Đổi biến model mong muốn ở cell cấu hình:
   ```python
   MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
   # Hoặc: "FacebookAI/xlm-roberta-base", "jhu-clsp/mmBERT-base", "google/rembert"...
   ```
4. Bấm **"Save Version" -> "Run All (Commit)"**. Notebook sẽ tự động train, lưu checkpoint Macro-F1 cao nhất và xuất ra `/kaggle/working/submission.csv`.

---

## 🔍 Checklist Kiểm Tra File Trước Khi Nộp Kaggle

Chạy kiểm tra nhanh để đảm bảo không bị lãng phí lượt nộp:
```python
from src.utils import validate_submission
validate_submission(sub_df)

# Kiểm tra phân bố nhãn (chuẩn: Benign ~60%, Jailbreak ~30%, Injection ~6-9%)
print(sub_df['label'].value_counts(normalize=True) * 100)
```
- [ ] Xây dựng giải pháp trên **đúng 1 model duy nhất** thuộc whitelist (không ensemble đa backbone).
- [ ] Khai báo `# Model: <tên_model>` ở dòng đầu tiên của notebook.
- [ ] File đủ **đúng 6.000 dòng**, 2 cột: `id` và `label`.
- [ ] Không chứa `NaN`, `null`, giá trị rỗng, hoặc mã số `0, 1, 2`.
- [ ] Lớp `PROMPT_INJECTION` đạt tỷ lệ từ **5% đến 10%** (nếu dưới 3% là mô hình bị lỗi, không được nộp).
- [ ] Đặt tên notebook nộp chuẩn: `<university>_<campus>_<team>_C1` (ví dụ: `RMIT_SGS_<team>_C1`).
