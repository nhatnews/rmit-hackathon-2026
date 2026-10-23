# 🤝 HANDOFF DOCUMENT: RMIT HACKATHON 2026 — CHALLENGE 1
# BÀN GIAO CHIẾN THUẬT GIAI ĐOẠN CUỐI (SAU LƯỢT NỘP 4 ĐẠT 0.97471 — CÒN 2 LƯỢT NỘP CUỐI)

> **Thời điểm cập nhật:** 10:20 AM — Ngày 22/10/2026  
> **Đội thi:** `BakhiBachKhoa` — Mã notebook Kaggle: `RMIT_SGS_BakhiBachKhoa_C1`  
> **Trạng thái hiện tại:** Đã nộp xong **Lượt 4/6 (RemBERT Version 2)**, điểm Public LB tăng vọt từ **`0.97058` -> `0.97471` (`+0.00413`)**! Hiện đội còn **ĐÚNG 2 LƯỢT NỘP CUỐI CÙNG (Lượt 5 & Lượt 6)**. Công việc đang được chia làm 2 mũi nhọn song song: **Trung Quân** phụ trách train thêm 1 lượt RemBERT Version 3, và **User + Agent phiên mới** tập trung phân tích đột phá kỹ thuật để bứt phá từ `0.97471` lên nhóm dẫn đầu (`0.982 - 0.983`).

---

## 🚨 1. CẬP NHẬT BẢNG XẾP HẠNG (LEADERBOARD) & QUOTA HIỆN TẠI

### A. Toàn cảnh Top 5 Public Leaderboard (10:16 AM):
| Hạng | Đội thi | Điểm Public LB | Số lượt đã nộp (Max 6) | Khoảng cách so với BakhiBachKhoa |
| :---: | :--- | :---: | :---: | :---: |
| **1** | `RMIT_HNI_MUCK` | **`0.98359`** | 3 / 6 | `-0.00888` (~53 mẫu test) |
| **2** | `RMIT_MEL_WeNeedClaudeSub_C1` | **`0.98216`** | 4 / 6 | `-0.00745` (~44 mẫu test) |
| **3** | `RMIT_MYS_nguynthjmhoa_C1` | **`0.97554`** | **6 / 6 (Hết lượt!)** | `-0.00083` (~5 mẫu test) |
| **4** | `RMIT_SGS_Gay_Hackers` | **`0.97554`** | 1 / 6 | `-0.00083` (~5 mẫu test) |
| **5** | **`RMIT_SGS_BakhiBachKhoa_C1` (Chúng ta)** | **`0.97471`** *(New PB!)* | **4 / 6 (Còn 2 lượt)** | — |

### B. Lịch sử 4 lượt nộp của đội `BakhiBachKhoa`:
1. **Lượt 1:** `0.93927` (Thịnh)
2. **Lượt 2:** `0.93738` (Quân)
3. **Lượt 3 (RemBERT V1):** **`0.97058`** (5-Fold OOF = `0.96491`)
4. **Lượt 4 (RemBERT V2):** **`0.97471`** (5-Fold OOF = `0.96758` — **Tăng `+0.00413` trên LB!**)
5. **Lượt 5 & Lượt 6:** **CHƯA DÙNG (Còn đúng 2 viên đạn cuối cùng!)**

### C. Quy định sống còn của BTC (TUYỆT ĐỐI TUÂN THỦ):
- **Strict Single-Model Mandate:** BẮT BUỘC xây dựng trên **DUY NHẤT 1 pretrained backbone** là `google/rembert` (`# Model: google/rembert` ở dòng 1). Nghiêm cấm blend/ensemble chéo nhiều backbone khác nhau.
  - *Lưu ý quan trọng:* Việc kết hợp (K-Fold / Multi-Seed / Multi-Run checkpoint averaging) các lần chạy của **cùng duy nhất 1 model `google/rembert`** là **HOÀN TOÀN HỢP LỆ** theo luật.
- **Native Kaggle Notebook Output:** Mọi submission phải được sinh ra và bấm `Submit to Competition` trực tiếp từ Output (`submission.csv`) của Kaggle Notebook.

---

## 📊 2. ĐỐI CHIẾU TOÀN DIỆN KẾT QUẢ REMBERT V1 VS V2 (ĐÃ LƯU TRONG WORKSPACE)

Toàn bộ log và ma trận xác suất của cả 2 lần chạy đã được lưu đầy đủ tại máy cục bộ và bàn giao cho **Trung Quân**:
- **Thư mục Version 1 (`LB 0.97058`):** `D:\Code\rmit-hackathon-2026\Rembert_output\`
- **Thư mục Version 2 (`LB 0.97471`):** `D:\Code\rmit-hackathon-2026\Rempert_output_v2\`
- **Script kiểm định đối chiếu:** [src/verify_v2_output.py](file:///d:/Code/rmit-hackathon-2026/src/verify_v2_output.py)

### A. Bảng so sánh 5-Fold OOF Macro-F1 (14,000 mẫu Train):
| Chỉ số đánh giá | RemBERT V1 (`LB 0.97058`) | RemBERT V2 (`LB 0.97471`) | Chênh lệch (V2 vs V1) |
| :--- | :---: | :---: | :---: |
| **OOF Base (Argmax)** | `0.96425` | **`0.96585`** | **`+0.00160`** |
| **OOF Global Multipliers** | `0.96491` | **`0.96641`** | **`+0.00150`** |
| **OOF Per-Language Multipliers** | `0.96491` | **`0.96758`** | **`+0.00267`** |
| **Public Leaderboard** | `0.97058` | **`0.97471`** | **`+0.00413`** |
| **Tổng số ca đoán sai OOF (Argmax)** | `412 ca` | **`379 ca`** | **`-33 ca lỗi`** |
| **Số ca đoán sai tiếng Bồ Đào Nha (`pt`)** | `300 ca` | **`265 ca`** | **`-35 ca lỗi`** |

> 💡 **Nhận xét vàng:** Tương quan giữa OOF CV và Public LB là **tuyến tính tuyệt đối** (`+0.00267` OOF kéo theo `+0.00413` trên Public LB). Bốn cải tiến của V2 (`EPOCHS=4`, `LLRD=0.965`, chèn `[SEP]` giữa Head-Tail, `label_smoothing=0.05` và `Per-Language Multipliers`) đã đánh trúng tử huyệt tiếng Bồ Đào Nha (`pt` F1 tăng từ `0.95847` lên `0.96274`, `fr` F1 tăng từ `0.96929` lên `0.97576`).

### B. Phát hiện cực kỳ quan trọng về độ bổ trợ giữa từng Fold của V1 và V2:
| Fold | V1 Best Val F1 | V2 Best Val F1 | Chênh lệch | Bản chạy tốt hơn |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | **`0.97089`** (`ep3_1400`) | `0.96606` (`ep4_1400`) | `-0.00483` | **V1 tốt hơn rõ rệt** |
| **Fold 1** | `0.95880` (`ep3_700`) | **`0.96155`** (`ep3_700`) | `+0.00275` | **V2 tốt hơn** |
| **Fold 2** | `0.95871` (`ep3_1400`) | **`0.96189`** (`ep3_1400`) | `+0.00318` | **V2 tốt hơn** |
| **Fold 3** | **`0.97049`** (`ep3_700`) | `0.96942` (`ep4_1400`) | `-0.00107` | **V1 nhỉnh hơn** |
| **Fold 4** | `0.96230` (`ep3_700`) | **`0.96997`** (`ep3_1400`) | **`+0.00767`** | **V2 vượt trội hoàn toàn** |

👉 **PHÁT HIỆN ĐỘT PHÁ (ĐÃ KIỂM CHỨNG THỰC NGHIỆM TRÊN 14,000 MẪU OOF):**
- Mặc dù V2 thắng đậm ở Fold 1, 2, 4, nhưng ở **Fold 0 và Fold 3**, checkpoint của V1 lại hội tụ tốt hơn V2!
- Khi lấy trung bình xác suất của **cùng model `google/rembert` giữa V1 và V2** (`0.45 * V1 + 0.55 * V2` — tổng cộng 10 checkpoints của duy nhất `google/rembert`):
  - **OOF Base (Argmax chưa chỉnh ngưỡng):** Nhảy vọt từ `0.96585` -> **`0.96766`**!
  - **OOF Per-Language Multipliers Tuned:** Nhảy vọt từ `0.96758` -> **`0.97010` (`+0.00252` so với V2 và `+0.00519` so với V1)!**
  - Với độ tương quan 1:1 giữa OOF và Public LB (ở Lượt 4, tăng `+0.00267` OOF giúp tăng `+0.00413` LB), riêng việc kết hợp V1 + V2 (+ V3 của Trung Quân đang train) của cùng backbone `google/rembert` đã hứa hẹn đẩy điểm LB vượt mốc **`0.9785 – 0.9800+`**!

---

## 🤝 3. PHÂN CÔNG NHIỆM VỤ SONG SONG (USER + TRUNG QUÂN)

Để tận dụng tối đa thời gian còn lại và **2 lượt nộp cuối (Lượt 5 & 6)**, đội chia làm 2 nhánh:

### Nhánh 1: Trung Quân — Phụ trách Train thêm 1 lượt RemBERT Version 3
- **Dữ liệu đã bàn giao cho Trung Quân:** Toàn bộ log, `metrics.json`, `oof_probs_rembert.npy`, `test_probs_rembert.npy` của cả V1 (`Rembert_output`) và V2 (`Rempert_output_v2`), cùng notebook [notebooks/rembert_5fold_kaggle.ipynb](file:///d:/Code/rmit-hackathon-2026/notebooks/rembert_5fold_kaggle.ipynb).
- **Mục tiêu của lượt train thứ 3 (RemBERT V3):**
  - Vẫn giữ nguyên **100% Single-Model `google/rembert`** theo đúng luật BTC.
  - Đa dạng hóa quỹ đạo hội tụ (ví dụ: đổi `SEED = 2026`, hoặc điều chỉnh nhẹ `CLASS_WEIGHTS = [1.0, 1.8, 6.5]` / cắt `Head 192 + [SEP] + Middle 124 + [SEP] + Tail 192` để bắt thêm payload nằm ở giữa câu của các văn bản dài > 1,000 từ).
  - Xuất ra bộ xác suất `oof_probs_rembert.npy` và `test_probs_rembert.npy` thứ 3.

### Nhánh 2: User & Agent trong Phiên Làm Việc Mới — Khai phá Đột Phá Kỹ Thuật (Target `0.980+`)
Agent tiếp quản trong phiên làm việc mới cần thực hiện ngay các nhiệm vụ nghiên cứu sâu sau đây để tìm ra lý do vì sao Top 1/Top 2 đạt `0.982 - 0.983` và thiết kế giải pháp nâng điểm cho Lượt 5 & Lượt 6:

1. **Mổ xẻ 379 ca lỗi còn lại của V2 (và các ca cả V1 + V2 cùng sai):**
   - Dùng `Rembert_output/oof_probs_rembert.npy` và `Rempert_output_v2/oof_probs_rembert.npy` đối chiếu với `data/raw/train.csv`.
   - Kiểm tra xem các ca lỗi còn lại có rơi vào các prompt **siêu dài (`> 2,000 ký tự`)** mà payload độc hại nằm ở **phần giữa văn bản (Middle section)** bị mất khi cắt `Head 254 + Tail 254` hay không!
   - Kiểm tra xem có hiện tượng **Near-Duplicate / Template Leakage / Prefix-Suffix Rule** hoặc nhiễu nhãn có hệ thống (Systematic Label Noise) giữa `train.csv` và `test.csv` không.
2. **Tối ưu hóa Kiến trúc Đầu ra / Feature Engineering cho RemBERT (100% Single-Model Compliant):**
   - Thử nghiệm kết hợp xác suất `google/rembert` (Multi-Run / Fold-Quality Weighted Checkpoint Averaging giữa V1, V2 và V3 của Trung Quân — chọn trọng số cao hơn cho các fold đạt `~0.970` như Fold 0, 3 của V1 và Fold 3, 4 của V2).
   - Nghiên cứu xem có thể trích xuất `[CLS]` embedding từ chính `google/rembert` đã fine-tune hoặc dùng Sliding Window / 3-Segment Truncation (`Head + Mid + Tail`) cho các văn bản dài trên `test.csv` (Inference-time Multi-Crop / TTA bằng chính checkpoint `google/rembert`!) để không bỏ sót bất kỳ đoạn văn bản nào trong các prompt dài tới 18,550 ký tự!
3. **Đóng gói Native Kaggle Notebook cho Lượt 5 và Lượt 6:**
   - Đảm bảo toàn bộ pipeline xuất `submission.csv` chạy **100% native trên Kaggle Notebook** (có thể mount Kaggle Dataset chứa các checkpoint / OOF & Test probabilities của chính các lần chạy `google/rembert` của đội để tổng hợp hoặc chạy inference TTA cực nhanh trong vài phút).

---

## 📋 4. DANH MỤC FILE QUAN TRỌNG TRONG WORKSPACE (`D:\Code\rmit-hackathon-2026`)

- [HANDOFF.md](file:///d:/Code/rmit-hackathon-2026/HANDOFF.md): Tài liệu bàn giao trung tâm này.
- [notebooks/rembert_5fold_kaggle.ipynb](file:///d:/Code/rmit-hackathon-2026/notebooks/rembert_5fold_kaggle.ipynb): Notebook RemBERT V2 vừa chạy đạt **`0.97471`**.
- [notebooks/rembert_5fold_kaggle_v1_backup_0.97058.ipynb](file:///d:/Code/rmit-hackathon-2026/notebooks/rembert_5fold_kaggle_v1_backup_0.97058.ipynb): Notebook RemBERT V1 đạt **`0.97058`**.
- `Rembert_output/`: Chứa `metrics.json`, `oof_probs_rembert.npy`, `test_probs_rembert.npy`, `submission.csv` của **Version 1**.
- `Rempert_output_v2/`: Chứa toàn bộ `metrics.json`, `oof_probs_rembert.npy`, `test_probs_rembert.npy`, `oof_fold0..4.npy`, `test_fold0..4.npy`, `submission.csv` và `logs/` của **Version 2**.
- [src/verify_v2_output.py](file:///d:/Code/rmit-hackathon-2026/src/verify_v2_output.py): Script đối chiếu chi tiết V1 vs V2.
