# 🏆 RMIT HACKATHON 2026 — CHALLENGE 1
# BÀI HỌC KINH NGHIỆM & HƯỚNG DẪN HUẤN LUYỆN DÀNH CHO TOÀN ĐỘI

> **Áp dụng cho:** Tất cả thành viên trong đội (Nhật Tân, Lê Cường Thịnh, Trung Quân, ...) khi huấn luyện bất kỳ model nào trong danh mục Approved Models (`RemBERT`, `XLM-RoBERTa`, `DeBERTa`, `mmBERT`, `Granite`, v.v.).
> **Mục tiêu:** Tối ưu hóa điểm số Macro-F1, tránh các sai lầm kỹ thuật chí mạng, và bảo vệ 4 lượt submit còn lại của toàn đội.

---

## 📌 1. BỐI CẢNH THỰC TẾ & TÌNH TRẠNG QUOTA HIỆN TẠI

1. **Quota Nộp Bài (Cực kỳ khẩn cấp):**
   - Ban tổ chức giới hạn **tối đa 6 lần nộp cho toàn bộ cuộc thi** trên toàn đội.
   - **Đã dùng 2 / 6 lượt:**
     - Lượt 1 (Thịnh): `FacebookAI/xlm-roberta-large` $\rightarrow$ **0.93927** (Tạm dẫn đầu đội).
     - Lượt 2 (Quân): `protectai/deberta-v3-base-prompt-injection-v2` $\rightarrow$ **0.93738**.
   - **CHỈ CÒN ĐÚNG 4 LƯỢT NỘP.**
   - 👉 **Nguyên tắc bất di bất dịch:** Chỉ được phép submit khi điểm **5-Fold OOF Macro-F1** trên máy vượt qua mốc cao nhất hiện tại (**$\ge 0.9395$**). Mọi lượt nộp phải qua Gatekeeper **Nhật Tân** kiểm duyệt.

2. **Luật Single-Model nghiêm ngặt (Strict Single-Model Mandate):**
   - BẮT BUỘC xây dựng giải pháp cuối cùng trên **DUY NHẤT 1 pretrained backbone** được duyệt.
   - **Nghiêm cấm tuyệt đối:** Không blend, ensemble hoặc stacking giữa các pretrained backbone khác nhau (ví dụ: cấm lấy prob của DeBERTa blend với XLM-RoBERTa). Vi phạm sẽ bị **Disqualify** ngay lập tức.
   - Được phép: K-fold checkpoint averaging của *cùng một* backbone model.

---

## 🔍 2. BỐN PHÁT HIỆN "VÀNG" RÚT RA TỪ SỐ LIỆU THỰC TẾ (DeBERTa vs XLM-RoBERTa)

Nhóm đã phân tích đối chiếu chi tiết 6,000 dòng test giữa 2 lần submit thực tế của Thịnh và Quân, rút ra các quy luật sau:

### Phát hiện 1: Tiếng Bồ Đào Nha (`pt`) là "Chiến trường quyết định" (Chiếm 70% bất đồng)
- **Độ đồng thuận tổng thể:** 2 model đồng thuận tới **95.13%** (5,708 / 6,000 mẫu).
- **Chỉ có 292 mẫu (4.87%) xảy ra bất đồng**, và chính 292 mẫu này quyết định thứ hạng:
  - Tiếng Anh (`en`): Đồng thuận **97.00%** (lệch 18 mẫu / 600)
  - Tiếng Pháp (`fr`): Đồng thuận **97.33%** (lệch 40 mẫu / 1,500)
  - Tiếng Ý (`it`): Đồng thuận **97.50%** (lệch 30 mẫu / 1,200)
  - **Tiếng Bồ Đào Nha (`pt`): Đồng thuận chỉ 92.44% (LỆCH TỚI 204 MẪU / 2,700!)**
- 👉 **Bài học:** Tiếng Bồ Đào Nha chiếm tới 45% tập dữ liệu. XLM-RoBERTa thắng điểm DeBERTa chủ yếu vì nó hiểu tiếng Bồ Đào Nha tốt hơn. Bất kỳ model nào train tiếp theo đều phải được kiểm tra kỹ F1 trên tập con tiếng Bồ Đào Nha.

### Phát hiện 2: Năng lực đa ngôn ngữ (Multilingual) quan trọng hơn Security Pretraining tiếng Anh
- `DeBERTa-prompt-injection` dù được pretrain chuyên về prompt injection nhưng dữ liệu tiền huấn luyện chủ yếu là tiếng Anh.
- Bộ dữ liệu cuộc thi này có tới **90% là ngôn ngữ Romance** (`pt`, `fr`, `it`) và chỉ có **10% tiếng Anh**.
- 👉 **Bài học:** Các mô hình đa ngôn ngữ lớn có từ vựng rộng (như `XLM-RoBERTa-large`, `RemBERT` với 250k token) có ưu thế vượt trội so với các mô hình chuyên security nhưng chỉ mạnh tiếng Anh.

### Phát hiện 3: Vùng tỷ lệ vàng của `PROMPT_INJECTION` là 6.5% – 7.5%
- `PROMPT_INJECTION` chỉ chiếm 7% số mẫu nhưng gánh tới **1/3 tổng điểm Macro-F1**.
- Phân phối nhãn thực tế:
  - DeBERTa (0.93738): 378 mẫu PROMPT_INJECTION (6.30%).
  - XLM-RoBERTa (0.93927): **393 mẫu PROMPT_INJECTION (6.55%)** (+15 mẫu).
- Đặc biệt, ở tiếng Pháp và tiếng Ý, DeBERTa đã bỏ sót 24 ca PROMPT_INJECTION (gán nhầm thành BENIGN) mà XLM-RoBERTa bắt được. 15 mẫu bắt thêm này chính là lý do trực tiếp giúp XLM-RoBERTa đạt điểm cao hơn.
- 👉 **Bài học:** Không bao giờ để model chỉ dùng `argmax()` chay nếu tỷ lệ dự đoán PROMPT_INJECTION < 6.0%. Bắt buộc dùng class-weights hoặc post-processing threshold/multipliers để đưa tỷ lệ nhãn này về **6.5% – 7.2% (khoảng 390 – 430 mẫu)**.

### Phát hiện 4: Prompt dài (Median 717 ký tự) là nơi model dễ phân vân nhất
- Các mẫu 2 model **đồng thuận**: độ dài ký tự trung vị là **266 ký tự**.
- Các mẫu 2 model **bất đồng**: độ dài ký tự trung vị là **717 ký tự** (dài gấp gần 3 lần!).
- Các prompt tấn công nguy hiểm thường ngụy trang bằng lời dẫn roleplay dài dòng ở đầu, và chèn lệnh tấn công/ký hiệu phân cách ở cuối.
- 👉 **Bài học:** Nếu chỉ cắt ngắn 512 token từ đầu văn bản (`head truncation`), model sẽ mất sạch payload tấn công ở đuôi! Bắt buộc phải áp dụng chiến lược **Head 255 + Tail 255**.

---

## 🛠️ 3. CHECKLIST KỸ THUẬT BẮT BUỘC KHI TRAIN BẤT KỲ MODEL NÀO

Mọi script huấn luyện (dù dùng mô hình nào) phải tuân thủ nghiêm ngặt 6 tiêu chuẩn sau:

### 1. Xử lý văn bản (Text Preprocessing)
- [ ] **Bảo tồn dấu Romance và Unicode:** Tuyệt đối không dùng regex phá hủy như `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. Dùng `unicodedata.normalize("NFC", text)` và loại bỏ null byte.
- [ ] **Bảo tồn chữ hoa/chữ thường (`cased`):** Giữ nguyên viết hoa để nhận diện các từ viết hoa trong tấn công prompt (như `IGNORE PREVIOUS INSTRUCTIONS`).
- [ ] **Thêm tiền tố ngôn ngữ (Language Prefix):** Luôn format input thành:
  ```python
  f"lang: {row['language']} | {clean_text}"
  ```
  giúp transformer kích hoạt các attention heads chuyên biệt cho ngôn ngữ đó.

### 2. Token Truncation (Head + Tail)
- [ ] Prompts trong bài thi có thể dài tới 18,550 ký tự. Không dùng `truncation=True` mặc định (chỉ lấy đầu).
- [ ] Áp dụng công thức cắt: **255 token đầu + 255 token cuối (+ CLS/SEP = 512 token)**:
  ```python
  if len(tokens) <= 510:
      ids = [cls_id] + tokens + [sep_id]
  else:
      ids = [cls_id] + tokens[:255] + tokens[-255:] + [sep_id]
  ```

### 3. Giải quyết mất cân bằng lớp (Class Imbalance cho 7% PROMPT_INJECTION)
- [ ] **Weighted Loss:** Sử dụng `CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 8.8]))` trong quá trình train.
- [ ] **Post-processing Multipliers:** Tối ưu hóa vector nhân xác suất `cand = [1.0, m1, m2]` trên toàn bộ ma trận OOF 14,000 dòng bằng Nelder-Mead hoặc Grid Search trước khi dự đoán tập test.

### 4. Smart Batching & Tối ưu VRAM (Cho GPU T4)
- [ ] **Smart Batching (Dynamic Padding):** Sắp xếp các câu có độ dài gần nhau vào cùng batch để giảm thiểu số lượng token padding thừa, giúp tăng tốc train 2–3 lần.
- [ ] **Gradient Checkpointing:** Bắt buộc bật `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` để tránh lỗi CUDA Out-Of-Memory (OOM).
- [ ] **Layer-wise Learning Rate Decay (LLRD):** Giảm learning rate ở các layer dưới và giữ learning rate cao hơn ở classifier head để tránh làm hỏng các biểu diễn ngôn ngữ đã pretrain.

### 5. Cơ chế Resume Checkpoint từng Fold
- [ ] Huấn luyện 5-Fold trên Kaggle có thể mất 2–4 tiếng. Hãy luôn lưu `oof_fold{k}.npy` và `test_fold{k}.npy` sau mỗi fold.
- [ ] Nếu phiên chạy bị ngắt hoặc rớt mạng ở Fold 3, cơ chế Resume sẽ nạp lại Fold 0, 1, 2 từ Output cũ và chỉ chạy tiếp các Fold còn lại.

---

## 📋 4. CHECKLIST PHÊ DUYỆT SUBMISSION (GATEKEEPER NHẬT TÂN)

Trước khi gửi bất kỳ file nào lên Kaggle Leaderboard, thành viên phải chạy đoạn script kiểm định sau:

```python
import pandas as pd
import numpy as np

sub = pd.read_csv("submission.csv")

# 1. Kích thước và cột
assert sub.shape == (6000, 2), f"Sai kích thước: {sub.shape}"
assert list(sub.columns) == ["id", "label"], f"Sai tên cột: {list(sub.columns)}"

# 2. Không có NaN / Null / Chuỗi rỗng
assert not sub.isna().any().any(), "Chứa giá trị NaN!"
assert (sub["label"].astype(str).str.len() > 0).all(), "Chứa chuỗi rỗng!"
assert not sub["id"].duplicated().any(), "Trùng lặp ID!"

# 3. Tập nhãn hợp lệ
VALID_LABELS = {"BENIGN", "JAILBREAK", "PROMPT_INJECTION"}
assert set(sub["label"].unique()) <= VALID_LABELS, "Nhãn không hợp lệ!"

# 4. Phân phối nhãn
dist = sub["label"].value_counts(normalize=True) * 100
pi_pct = dist.get("PROMPT_INJECTION", 0.0)
print("PHÂN PHỐI DỰ ĐOÁN:")
print(dist)
assert 5.0 <= pi_pct <= 9.0, f"Cảnh báo: Tỷ lệ PROMPT_INJECTION ({pi_pct:.2f}%) nằm ngoài vùng an toàn 5%–9%!"
```

### Tiêu chuẩn cấp phép nộp bài (Chỉ 4 lượt còn lại):
1. [ ] **Single Backbone:** Xác nhận 100% chỉ sử dụng DUY NHẤT 1 approved backbone (không blend backbone chéo).
2. [ ] **Dòng 1 Notebook:** Đã ghi rõ `# Model: <tên_approved_model>`.
3. [ ] **OOF Macro-F1:** Điểm validation 5-fold trên máy phải **$\ge 0.9395$** (đủ tiềm năng phá kỷ lục hiện tại 0.93927 của XLM-RoBERTa).
4. [ ] **Đồng thuận với Baseline Ensemble:** Độ đồng thuận với 5,708 mẫu đồng thuận của Thịnh & Quân phải đạt **$\ge 95\%$**.
5. [ ] **Được Nhật Tân duyệt** và ghi nhận vào nhật ký submission.

---
*Tài liệu được tổng hợp từ dữ liệu đối soát thực tế RMIT Hackathon 2026 — Đội BakhiBachKhoa.*
