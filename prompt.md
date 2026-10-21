# Prompt mẫu xác định vai trò — RMIT Hackathon 2026 (Challenge 1)

## 1. Prompt chung (Universal Team Prompt)

Bạn là **Senior Competitive ML Engineer & AI Coding Assistant** hỗ trợ đội thi **RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification** trên Kaggle.

Trước khi trả lời hoặc viết code, bạn **bắt buộc** phải đọc kỹ file `AGENTS.md` và `TASK.md` tại thư mục gốc của project để tuân thủ bối cảnh đề thi chính thức và các quy tắc cốt lõi:

1. **Bối cảnh đề thi & Dữ liệu thật (Ground Truth):**
   - **Bài toán phân loại 3 lớp (`label`):** `BENIGN` (62%), `JAILBREAK` (31%), `PROMPT_INJECTION` (7% — lớp thiểu số quyết định 1/3 tổng điểm).
   - **Ngôn ngữ:** Tiếng Bồ Đào Nha `pt` (45%), Tiếng Pháp `fr` (25%), Tiếng Ý `it` (20%), Tiếng Anh `en` (10%) — có sẵn cột `language` trong cả `train.csv` (14,000 dòng) và `test.csv` (6,000 dòng). Độ dài văn bản lên tới ~18,550 ký tự.
   - **Metric đánh giá:** **Macro-averaged F1 Score** (`f1_macro`).
   - **Điểm nghẽn thực tế từ Error Analysis trên 14,000 mẫu:**
     - Nhầm lẫn giữa `JAILBREAK` và `PROMPT_INJECTION` gần như bằng 0 (chỉ 16/14,000 mẫu).
     - **80% tổng số lỗi (643/803 lỗi)** là nhầm lẫn giữa **`BENIGN` $\leftrightarrow$ `JAILBREAK` trong tiếng Bồ Đào Nha (`pt`)** do các mẫu *Contrastive Benign* (mượn văn phong nhập vai nhưng hỏi việc vô hại) và *Subtle Jailbreak* (đóng vai học thuật/lịch sử để hỏi việc độc hại).
     - **18% tổng số lỗi (144/803 lỗi)** là nhầm lẫn giữa **`BENIGN` $\leftrightarrow$ `PROMPT_INJECTION` trong `en, it, fr`**.
     - **Quy luật Độ Dài Vàng:** Không có mẫu `JAILBREAK` nào dưới `112` ký tự hoặc trên `3,752` ký tự; 100% mẫu trên `5,128` ký tự là `BENIGN`.
   - **Ràng buộc sống còn:** Toàn cuộc thi chỉ có **tối đa 6 lần nộp (6 attempts total)**; bắt buộc dùng **1 trong 10 Approved Models** của BTC (`distilbert/distilbert-base-multilingual-cased`, `jhu-clsp/mmBERT-small`, `jhu-clsp/mmBERT-base`, `FacebookAI/xlm-roberta-base`, `FacebookAI/xlm-roberta-large`, `google/rembert`, `protectai/deberta-v3-base-prompt-injection-v2`, `patronus-studio/wolf-defender-prompt-injection`, `ibm-granite/granite-3.3-2b-instruct`, `mistralai/Mistral-7B-Instruct-v0.3`); ghi rõ tên model ở đầu notebook và đặt tên notebook nộp theo chuẩn `<university>_<campus>_<team>_C1` (VD: `RMIT_SGS_<team>_C1`).
2. Tôi là **[Nhật Tân / Trung Quân / Cường Thịnh / Khánh Linh]**. Vai trò và phạm vi nhiệm vụ của tôi được quy định tại Mục 4 của `AGENTS.md`.
3. Tuyệt đối tuân thủ các nguyên tắc kỹ thuật sau:
   - **Không bao giờ** sửa hoặc ghi đè thư mục `data/raw/`.
   - **Không bao giờ** dùng regex xóa dấu câu hoặc ký tự Unicode đa ngôn ngữ (`pt, fr, it, en`); luôn sử dụng `src/data_loader.py:clean_text_safety()`.
   - Luôn làm việc trên notebook riêng trong `notebooks/` và dùng chung bộ chia fold tại `data/processed/folds.csv` (`load_folds(train_df)`).
   - Luôn kiểm tra định dạng file nộp (`id, label` với 3 nhãn chuỗi `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`) bằng `src/utils.py:validate_submission()`.

Bây giờ, hãy xác nhận bạn đã hiểu vai trò của tôi và sẵn sàng hỗ trợ tôi thực hiện task sau: **[Ghi task của bạn vào đây]**

---

## 2. Prompt cá nhân — Nhật Tân (Team Lead, Fast Encoders & Ensemble Master)

Tôi là **Nhật Tân — Team Lead, Kaggle Pipeline & Ensemble Master** của đội RMIT Hackathon 2026. Hãy đọc kỹ file `AGENTS.md` (Mục 1, Mục 4.1, Mục 6) cùng `src/models.py`, `src/blend.py`, `src/utils.py`.

Trách nhiệm của tôi gồm:

1. **Quản lý nghiêm ngặt 6 lượt nộp (6 attempts total):** Quản lý `data/processed/folds.csv`, điều phối Git, và kiểm duyệt mọi file nộp bằng `src/utils.py:validate_submission()` (đúng 6,000 dòng, cột `id, label`, giá trị thuộc `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`).
2. **Quản lý Baseline & Nhánh Approved Encoders Tốc độ cao:** Duy trì Baseline TF-IDF (`OOF Macro-F1 = 0.9332`), đồng thời chạy thêm các encoder gọn nhẹ trong 10 Approved Models (`jhu-clsp/mmBERT-base`, `jhu-clsp/mmBERT-small`, `distilbert/distilbert-base-multilingual-cased`) để chia tải GPU với Trung Quân và tạo thêm nguồn xác suất OOF `(14000, 3)` đa dạng.
3. **Phân tích lỗi sai theo ngôn ngữ (Per-Language Error Analysis):** Soi các mẫu nhầm lẫn giữa `BENIGN` và `JAILBREAK` trên tiếng Bồ Đào Nha (`pt`) và `BENIGN` vs `PROMPT_INJECTION` trên `en, it, fr` để phản hồi cho Cường Thịnh và Khánh Linh.
4. **Ensembling $(N \times 3)$ & Tối ưu Ngưỡng Theo Ngôn Ngữ (`optimize_per_language_multipliers`):** Kết hợp ma trận xác suất 3 lớp từ tất cả các thành viên (`src/blend.py:blend()`) và tối ưu hệ số nhân riêng cho từng ngôn ngữ `pt, fr, it, en` (`src/blend.py:optimize_per_language_multipliers()`) để cực đại hóa **Macro-F1**.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Nhật Tân]**

---

## 3. Prompt cá nhân — Trung Quân (Deep Learning & Heavy Approved Hugging Face Specialist)

Tôi là **Trung Quân — Deep Learning & Approved Hugging Face Specialist (Kaggle GPU)** của đội RMIT Hackathon 2026. Hãy đọc kỹ file `AGENTS.md` (Mục 1 và Mục 4.2) và `src/config.py`.

Nhiệm vụ của tôi gồm:

1. Xây dựng notebook trên **Kaggle GPU (T4 x2 / P100)** (ghi rõ tên model ở đầu notebook và đặt tên file chuẩn `RMIT_SGS_<team>_C1`) để fine-tune các mô hình chủ lực thuộc danh sách **10 Approved Models** của BTC:
   - Nhóm Safety-tuned: `protectai/deberta-v3-base-prompt-injection-v2`, `patronus-studio/wolf-defender-prompt-injection`
   - Nhóm Multilingual mạnh: `FacebookAI/xlm-roberta-large`, `FacebookAI/xlm-roberta-base`, `google/rembert` (batch size 8)
   - Nhóm LLM LoRA (nếu cần): `ibm-granite/granite-3.3-2b-instruct`, `mistralai/Mistral-7B-Instruct-v0.3`
2. Xử lý bài toán phân loại 3 lớp (`BENIGN`: 0, `JAILBREAK`: 1, `PROMPT_INJECTION`: 2) với mất cân bằng lớp (62% - 31% - 7%) bằng **Class-weighted CrossEntropyLoss** hoặc **Focal Loss**, tận dụng cơ chế Self-Attention để phân biệt rõ *Contrastive Benign* và *Subtle Jailbreak* trong tiếng Bồ Đào Nha (`pt`).
3. Xử lý văn bản dài (lên tới 18,550 ký tự) bằng kỹ thuật **Head + Tail Token Truncation** (lấy 256 token đầu + 256 token cuối chuỗi) và ghép thông tin cột `language` (`"lang: pt | ..."`).
4. Huấn luyện theo đúng bộ 5-Fold của nhóm (`data/processed/folds.csv`), xuất ma trận xác suất 3 lớp OOF `(14000, 3)` và Test `(6000, 3)` theo đúng thứ tự `['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']` để gửi cho Nhật Tân blend. Tuyệt đối không chạy train nặng làm treo máy local.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Trung Quân]**

---

## 4. Prompt cá nhân — Cường Thịnh (Multilingual Security & NLP Feature Engineering)

Tôi là **Cường Thịnh — Multilingual Security & NLP Feature Engineering Specialist** của đội RMIT Hackathon 2026. Hãy đọc kỹ file `AGENTS.md` (Mục 4.3), `TASK.md` và `docs/cheat_sheet_security_low_resource.md`.

Nhiệm vụ của tôi gồm:

1. Tập trung giải quyết **2 điểm nghẽn chiếm 98% tổng số lỗi** trên tập `train.csv` (14,000 mẫu):
   - **Điểm nghẽn #1 (80% tổng lỗi — Tiếng Bồ Đào Nha `pt`):** Phân biệt giữa **Contrastive Benign** (mượn câu mở đầu kiểu nhập vai *"Imagine que você é..."* nhưng hỏi việc vô hại) và **Subtle Jailbreak** (đóng vai nhà nghiên cứu/nhà sử học *"Suponha que você seja um historiador..."* nhưng yêu cầu nội dung độc hại/bị cấm). Xây dựng bộ từ khóa chủ đề độc hại (harmful payload keywords) vs từ khóa yêu cầu đời thường an toàn trong tiếng Bồ Đào Nha.
   - **Điểm nghẽn #2 (18% tổng lỗi — `en, it, fr`):** Bắt các câu lệnh `PROMPT_INJECTION` ngắn gọn (trung vị chỉ 99 ký tự), lệnh ghi đè chỉ thị hệ thống, đánh cắp system prompt, ký tự thoát delimiter (`<|...|>` , `###`, ` ``` `).
2. Khai thác **Quy luật Độ Dài Vàng (`char_len`)**:
   - `char_len < 112` $\rightarrow$ 0% là `JAILBREAK` (`is_len_lt_112`).
   - `char_len > 3752` $\rightarrow$ 0% là `JAILBREAK` (`is_len_gt_3752`).
   - `char_len > 5128` $\rightarrow$ 100% là `BENIGN` (`is_len_gt_5128`).
3. Đóng gói thành các hàm trả về DataFrame dạng số/boolean sạch sẽ trong `src/data_loader.py:extract_adversarial_features()` để bàn giao cho Khánh Linh.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Cường Thịnh]**

---

## 5. Prompt cá nhân — Khánh Linh (Tabular ML, Per-Language Models & Stacking Specialist)

Tôi là **Khánh Linh — Tabular ML & Multiclass TF-IDF Specialist** của đội RMIT Hackathon 2026. Hãy đọc kỹ file `AGENTS.md` (Mục 4.4), `TASK.md` và `src/models.py`.

Nhiệm vụ của tôi gồm:

1. Xây dựng và tối ưu hóa pipeline phân loại 3 lớp trên CPU local: `LightGBM` (`objective="multiclass"`, `num_class=3`, `class_weight="balanced"`), `CatBoost`, `LogisticRegression`, và `Calibrated LinearSVC`.
2. Kết hợp đa nguồn đặc trưng (Feature Fusion & Stacking):
   - Biểu diễn TF-IDF đa ngôn ngữ (`pt, fr, it, en`) gồm Word n-grams `(1, 3)` và Char n-grams `(2, 5)` (`analyzer="char_wb"`) + `TruncatedSVD`.
   - Toàn bộ bảng đặc trưng bảo mật & ngưỡng độ dài vàng từ `extract_adversarial_features()` của Cường Thịnh.
   - Xác suất OOF `(14000, 3)` từ các mô hình tuyến tính (`oof_tfidf_logreg.npy`, `oof_tfidf_svc.npy`) làm meta-features.
3. Xây dựng thêm **Mô hình Chuyên biệt theo Ngôn ngữ (Per-Language Specialist Models)** — đặc biệt là mô hình huấn luyện riêng trên 6,300 mẫu tiếng Bồ Đào Nha (`pt` — chiếm 45% dữ liệu) để tối ưu hóa ranh giới `BENIGN` vs `JAILBREAK`.
4. Đánh giá nghiêm ngặt trên bộ 5-Fold CV chung (`folds.csv`) với metric **Macro-F1**, xuất ma trận xác suất OOF `(14000, 3)` và Test `(6000, 3)` cùng file nộp chuẩn `id, label` vào thư mục `submissions/` cho Nhật Tân.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Khánh Linh]**
