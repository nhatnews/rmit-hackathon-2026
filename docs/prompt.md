# Prompt mẫu xác định vai trò — RMIT Hackathon 2026

## 1. Prompt chung (Universal Team Prompt)

Bạn là AI Coding Assistant hỗ trợ đội thi RMIT Hackathon 2026 (chủ đề: Security x Generative AI x Low-Resource Languages).

Trước khi trả lời hoặc viết code, bạn **bắt buộc** phải đọc kỹ file `AGENTS.md` tại thư mục gốc của project để tuân thủ các quy tắc cốt lõi:

1. Tôi là **[Nhật Tân / Trung Quân / Cường Thịnh / Khánh Linh]**. Vai trò và phạm vi nhiệm vụ của tôi được quy định tại Mục 4 của `AGENTS.md`.
2. Mọi giải pháp, gợi ý code và câu trả lời của bạn phải bám sát vai trò của tôi, không vượt quá phạm vi hoặc làm phân tán nhiệm vụ của các thành viên khác.
3. Tuyệt đối tuân thủ các nguyên tắc sau:
   - **Không bao giờ** sửa hoặc ghi đè thư mục `data/raw/`.
   - **Không bao giờ** dùng regex để xóa dấu tiếng Việt hoặc ký tự đặc biệt; sử dụng `clean_text_safety()`.
   - **Không bao giờ** hardcode API keys hoặc tokens vào code.
   - Luôn làm việc trên notebook riêng trong `notebooks/`.
   - Luôn kiểm tra định dạng file nộp bằng `src/utils.py:validate_submission()`.

Bây giờ, hãy xác nhận bạn đã hiểu vai trò của tôi và sẵn sàng hỗ trợ tôi thực hiện task sau: **[Ghi task của bạn vào đây]**

---

## 2. Prompt cá nhân — Nhật Tân (Team Lead, Kaggle Pipeline & Active Modeler)

Tôi là Nhật Tân — Team Lead, Kaggle Pipeline Master & Active Modeler của đội RMIT Hackathon 2026. Hãy đọc file `AGENTS.md` (đặc biệt là Mục 4.1 và Mục 6).

Trách nhiệm của tôi gồm:

1. Quản lý luồng nộp bài lên Kaggle (5 lần/ngày), kiểm duyệt file bằng `src/utils.py:validate_submission()`, và điều phối merge Git.
2. Xây dựng và nộp bản Baseline đầu tiên (10:00 - 11:30) bằng TF-IDF + LogisticRegression / LinearSVC để ăn chắc điểm sớm cho cả team.
3. Phát triển nhánh mô hình thứ 3 bổ trợ: Multilingual Sentence Embeddings (`sentence-transformers` / API embeddings) hoặc Linear / CatBoost để đa dạng hóa ensemble.
4. Phân tích lỗi sai (Error Analysis / Hard Examples): Soi các mẫu false positive / false negative để phát hiện các kỹ thuật lẩn tránh mới và báo cho Cường Thịnh bổ sung regex, Khánh Linh tinh chỉnh features.
5. Tối ưu hóa Ensembling / Blending: Hiệu chuẩn xác suất (Platt scaling / Isotonic), Rank Averaging và tối ưu trọng số bằng Scipy/Optuna trên OOF predictions.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Nhật Tân]**

---

## 3. Prompt cá nhân — Trung Quân (Deep Learning & Hugging Face)

Tôi là Trung Quân — Deep Learning & Hugging Face Specialist của đội RMIT Hackathon 2026. Hãy đọc file `AGENTS.md` (đặc biệt là Mục 4.2).

Nhiệm vụ của tôi là xây dựng notebook fine-tune các mô hình Pretrained Transformers đa ngôn ngữ (mDeBERTa-v3, XLM-RoBERTa, PhoBERT) trên Kaggle Notebooks GPU (T4 x2). Tuyệt đối không gợi ý train nặng làm đơ máy local. Kết quả dự đoán phải xuất ra dạng xác suất (probabilities).

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Trung Quân]**

---

## 4. Prompt cá nhân — Cường Thịnh (Security & NLP Feature Engineering)

Tôi là Cường Thịnh — Security & NLP Feature Engineering Specialist của đội RMIT Hackathon 2026. Hãy đọc file `AGENTS.md` (đặc biệt là Mục 4.3 và file `docs/cheat_sheet_security_low_resource.md`).

Nhiệm vụ của tôi là bóc tách các đặc trưng tấn công bảo mật (jailbreak, DAN personas, chuỗi Base64/Hex, teencode tiếng Việt, ký tự cách nhau, Unicode NFC) thành các hàm trả về numeric/boolean features để bàn giao cho Khánh Linh.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Cường Thịnh]**

---

## 5. Prompt cá nhân — Khánh Linh (Machine Learning & TF-IDF / Tabular)

Tôi là Khánh Linh — Machine Learning & TF-IDF / Tabular Specialist của đội RMIT Hackathon 2026. Hãy đọc file `AGENTS.md` (đặc biệt là Mục 4.4).

Nhiệm vụ của tôi là tối ưu hóa TF-IDF (word + char n-grams 2–5), kết hợp các features của Cường Thịnh vào mô hình LightGBM / CatBoost / LogisticRegression, thiết lập 5-Fold Stratified CV trên CPU local và xuất file dự đoán chuẩn format ra thư mục `submissions/` cho Nhật Tân nộp.

Hãy hỗ trợ tôi theo đúng vai trò này cho task sau: **[Nhập task của Khánh Linh]**
