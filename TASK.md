# 🎯 KẾ HOẠCH TÁC CHIẾN & PHÂN CÔNG NHIỆM VỤ — RMIT HACKATHON 2026
> **Challenge 1: Multilingual Prompt-Attack Classification**

---

# 🟢 PHẦN 1: TÓM TẮT NHANH CHO CẢ ĐỘI

### 📌 Đề bài thực tế:
* **Dữ liệu vào:** 14.000 câu văn bản (`train.csv`) gồm 4 thứ tiếng: Bồ Đào Nha (`pt` - 45%), Pháp (`fr` - 25%), Ý (`it` - 20%), Anh (`en` - 10%).
* **Nhiệm vụ:** Chọn 1 mô hình BTC cho phép $\rightarrow$ Train mô hình $\rightarrow$ Đoán 3 nhãn cho 6.000 câu test (`test.csv`):
  * `BENIGN` (62%): An toàn.
  * `JAILBREAK` (31%): Đóng vai lừa AI vượt rào.
  * `PROMPT_INJECTION` (7% - **lớp quyết định điểm số**): Chiếm quyền điều khiển hệ thống.
* **Chấm điểm:** **Macro-F1** (3 nhãn quan trọng ngang nhau, nhãn 7% tính điểm bằng nhãn 62%).
* **Ràng buộc:** Toàn bộ cuộc thi chỉ được **nộp tối đa 6 lần**.

---

### 🗺️ Luồng 4 bước tác chiến của 4 thành viên

```
[Bước 1: Làm sạch Text] ──> [Bước 2: Chạy Train GPU] ──> [Bước 3: Chỉnh Phạt & Đánh Giá] ──> [Bước 4: Nộp Bài]
    (CƯỜNG THỊNH)                (TRUNG QUÂN)                   (KHÁNH LINH)                  (NHẬT TÂN)
```

| Thành viên | Vai trò phụ trách | Việc chính cần làm | Sản phẩm đầu ra |
| :--- | :--- | :--- | :--- |
| **1. CƯỜNG THỊNH** | **Dữ liệu & Chữ nghĩa** | - Giữ nguyên dấu tiếng Pháp/Bồ Đào Nha/Ý, không xóa bừa.<br>- Ghép tiền tố: `lang: <tiếng> \| <nội dung câu>`.<br>- Cắt câu dài (giữ đoạn đầu + đoạn cuối). | Hàm xử lý text nạp thẳng vào notebook. |
| **2. TRUNG QUÂN** | **Kaggle GPU & Model** | - Mở Kaggle Notebook GPU (T4 x2), đặt tên chuẩn `RMIT_SGS_<team>_C1`.<br>- Tải model: `protectai/deberta-v3-base-prompt-injection-v2`.<br>- Bấm chạy train và cho đoán tập test. | Notebook chạy mượt mà, xuất ra `submission.csv`. |
| **3. KHÁNH LINH** | **Chỉnh phạt & Đánh giá** | - Cài công thức phạt nặng nếu đoán sai nhãn 7% (phạt x8 lần).<br>- Chỉnh tốc độ học và số vòng (3-4 epochs) để model không học vẹt.<br>- Canh điểm Macro-F1 trên 5 phần dữ liệu chia nhỏ. | Bộ tham số tối ưu + Checkpoint thông minh nhất. |
| **4. NHẬT TÂN** | **Tối ưu cuối & Nộp bài** | - Giữ chìa khóa 6 lượt nộp (duyệt file bằng `validate_submission`).<br>- Tối ưu ngưỡng xác suất phút chót để kích thêm điểm Macro-F1.<br>- Trực tiếp submit lên bảng xếp hạng Kaggle. | 6 lượt nộp chuẩn xác, tối đa hóa thứ hạng. |

---

# 🔵 PHẦN 2: HÀNG RÀO KỸ THUẬT CHO AI AGENT (GEMINI / ANTIGRAVITY)
> *Các thành viên copy phần này vào prompt cho AI của mình để AI không đi lạc và không vi phạm quy chế thi.*

### 🚨 4 Ràng buộc kỹ thuật bất di bất dịch (Zero Exceptions)
1. **Chỉ dùng 1 model nòng cốt trong Whitelist BTC:** Dồn 100% sức cho **`protectai/deberta-v3-base-prompt-injection-v2`**. Dòng đầu tiên của Kaggle Notebook phải ghi rõ:
   ```python
   # Model: protectai/deberta-v3-base-prompt-injection-v2
   ```
2. **Tuyệt đối không xóa dấu Unicode và ký tự tấn công:** Tuyệt đối không dùng `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. Phải giữ nguyên dấu Romance (`é, è, ê, ç, ã, õ, á, í...`) và ký tự markdown (` ``` `, `<|...|>`). Luôn dùng hàm [`clean_text_safety()`](file:///d:/Code/rmit-hackathon-2026/src/data_loader.py).
3. **Cả 4 người dùng chung 1 bộ chia Fold:** Mọi đánh giá CV phải đọc từ [`data/processed/folds.csv`](file:///d:/Code/rmit-hackathon-2026/data/processed/folds.csv) (`load_folds(train_df)`), cấm tự ý chia fold bằng seed ngẫu nhiên khác.
4. **Bảo vệ Quota 6 lần nộp:** File nộp phải đủ **đúng 6.000 dòng**, 2 cột `id, label` (giá trị chuỗi `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`), không được chứa số `0, 1, 2` hay `NaN`. Luôn kiểm tra bằng [`validate_submission()`](file:///d:/Code/rmit-hackathon-2026/src/utils.py).

---

### 📋 Nhiệm vụ kỹ thuật chi tiết của từng thành viên

#### 1. Cường Thịnh — Prompt Formatting & Text Engineering
* **Ghép tiền tố ngôn ngữ:** Định dạng đầu vào transformer:
  ```python
  formatted_text = f"lang: {language} | {clean_text_safety(text)}"
  ```
* **Xử lý văn bản siêu dài (Head + Tail Truncation):** Văn bản dài tới 18.550 ký tự trong khi DeBERTa chỉ nhận 512 tokens. Cắt lấy **256 token đầu + 256 token cuối** (lệnh tấn công thường giấu ở cuối văn bản).
* **Quy luật độ dài vàng (Golden Length Rules):**
  * `char_len < 112` hoặc `char_len > 3752` $\rightarrow$ Chắc chắn **0% là `JAILBREAK`**.
  * `char_len > 5128` $\rightarrow$ **100% là `BENIGN`**.
* **Giải quyết 80% lỗi ở Tiếng Bồ Đào Nha (`pt` - 45% data):** Phân biệt giữa *Contrastive Benign* (đóng vai giả định nhưng hỏi việc an toàn) và *Subtle Jailbreak* (giả danh khoa học/lịch sử để hỏi việc độc hại).

#### 2. Trung Quân — Kaggle GPU & Model Training Lead
* **Khởi tạo Notebook chuẩn BTC:**
  * Bật GPU Kaggle (T4 x2), đặt tên notebook: `<university>_<campus>_<team>_C1` (VD: `RMIT_SGS_<team>_C1`).
  * Khai báo model ở dòng 1: `# Model: protectai/deberta-v3-base-prompt-injection-v2`.
  * Giữ notebook ở chế độ **Private** trong suốt thời gian thi đấu.
* **Pipeline Huấn luyện với Hugging Face:**
  * Sử dụng `AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2", num_labels=3)`.
  * Ánh xạ nhãn: `0: BENIGN, 1: JAILBREAK, 2: PROMPT_INJECTION`.
  * Bật mixed precision (`fp16=True`) để tiết kiệm bộ nhớ GPU và tăng tốc train gấp đôi.
* **Đảm bảo tính tái lập (End-to-End Reproducibility):**
  * Đảm bảo notebook có thể bấm **"Save & Run All (Commit)"** và tự động xuất ra file `/kaggle/working/submission.csv` mà không bị gián đoạn.

#### 3. Khánh Linh — Hyperparameters, Loss Tuning & Validation Specialist
* **Xử lý mất cân bằng lớp 62% - 31% - 7% (Class-Weighted Loss):**
  * Áp dụng trọng số phạt trong PyTorch để phạt nặng khi đoán sai lớp thiểu số `PROMPT_INJECTION`:
    ```python
    weights = torch.tensor([1.0, 2.0, 8.8]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    ```
* **Dò siêu tham số tối ưu (Hyperparameters):**
  * Learning rate: `1.5e-5` – `2e-5` kết hợp Cosine scheduler warmup 10%.
  * Batch size hiệu dụng: `16` – `32` (dùng `gradient_accumulation_steps=2` nếu batch size thực là 8).
  * Số epochs: `3` – `4` epochs (hạn chế train nhiều gây overfit). Weight decay: `0.01`.
* **Quản trị 5-Fold Validation:**
  * Theo dõi điểm **OOF Macro-F1** qua từng epoch trên đúng file `folds.csv`.
  * Luôn chọn checkpoint có điểm **Macro-F1** cao nhất (không chọn theo validation loss).

#### 4. Nhật Tân — Team Lead, Kaggle Pipeline & Quota Master
* **Quản trị nghiêm ngặt Quota 6 lần nộp:**
  * Lần #1 nộp ngay file [`submissions/baseline_tfidf_blend.csv`](file:///d:/Code/rmit-hackathon-2026/submissions/baseline_tfidf_blend.csv) (OOF = `0.9316`) để lấy điểm mốc và kiểm tra hệ thống.
  * Giữ chặt 5 lần nộp còn lại cho các cải tiến của DeBERTa-v3.
* **Tối ưu ngưỡng xác suất phút chót (Macro-F1 Threshold Tuning):**
  * Tích hợp hàm nhân hệ số xác suất $(m_0, m_1, m_2)$ từ [`src/blend.py`](file:///d:/Code/rmit-hackathon-2026/src/blend.py) vào cell cuối của notebook:
    ```python
    # Tăng độ nhạy cho lớp 7% PROMPT_INJECTION
    probs_adjusted = probs * np.array([m_benign, m_jailbreak, m_injection])
    preds = np.argmax(probs_adjusted, axis=1)
    ```
* **Kiểm duyệt bài nộp:** Chạy [`validate_submission()`](file:///d:/Code/rmit-hackathon-2026/src/utils.py) kiểm tra file xuất ra trước khi bấm nộp trên Kaggle.

---

# 📅 PHẦN 3: PHÂN BỔ 6 LƯỢT NỘP BÀI (KAGGLE SUBMISSION QUOTA)

| Lượt nộp | Thời điểm dự kiến | Nội dung bài nộp | Mục tiêu Macro-F1 | Người phụ trách duyệt |
| :---: | :---: | :--- | :---: | :---: |
| **#1** | Đầu buổi sáng (10:30) | `baseline_tfidf_blend.csv` (Có sẵn trong repo) | **`0.9316`** (Mở hàng + thông kết nối) | **Nhật Tân** |
| **#2** | Giữa buổi sáng (11:30) | DeBERTa-v3 Fold đầu tiên trên Kaggle GPU | **`0.9500+`** (Kiểm tra sức mạnh Deep Learning) | Nhật Tân / Trung Quân |
| **#3** | Đầu buổi chiều (13:30) | DeBERTa-v3 + Class Weights + Tiền tố `lang:` | **`0.9600+`** (Cứu điểm lớp 7% & đa ngôn ngữ) | Cả nhóm |
| **#4** | Giữa buổi chiều (15:00) | DeBERTa-v3 + Head-Tail 512 + Độ dài vàng | **`0.9650+`** (Bắt prompt dài 18k ký tự) | Cả nhóm |
| **#5** | Cuối chiều (16:30) | Checkpoint tốt nhất 5-Fold + Tối ưu ngưỡng $(m_0, m_1, m_2)$ | **`0.9700+`** (Chốt điểm tranh giải) | Cả nhóm |
| **#6** | 17:30 | **LƯỢT DỰ PHÒNG KHẨN CẤP** (Chỉ dùng khi có cải tiến đột phá) | Top 1 Leaderboard | **Nhật Tân** |
