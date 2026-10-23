# Hướng Dẫn Huấn Luyện `google/rembert` 5-Fold Qua Đêm Trên Kaggle

Tài liệu hướng dẫn chi tiết từng bước dành cho các thành viên trong đội thi để chạy notebook `notebooks/rembert_5fold_kaggle.ipynb` trên hạ tầng Kaggle GPU T4 x2.

---

## 1. Tổng Quan & Luật Thi Single-Model

- **Backbone sử dụng:** DUY NHẤT `google/rembert` (32 layers, hidden dimension 1152, ~575M parameters).
- **Luật tối thượng:** Theo quy định của BTC, toàn bộ giải pháp nộp phải được xây dựng trên **chính xác 1 pretrained backbone duy nhất**. Tuyệt đối **KHÔNG** kết hợp, ensemble, hay blend với các backbone khác (như TF-IDF, DeBERTa, XLM-RoBERTa,...). Việc lấy trung bình xác suất từ 5 fold checkpoint của cùng một backbone RemBERT là hợp lệ theo quy chế.
- **Tên Notebook quy định:** `RMIT_SGS_BakhiBachKhoa_C1`.
- **Mục tiêu:** Thu được `submission.csv` và `metrics.json` (OOF Macro-F1 trên 5-fold chuẩn) để so sánh chọn ra ứng viên Single Champion Model tốt nhất của team.

---

## 2. Chuẩn Bị Dataset `rmit-folds` Trên Kaggle (3 Phút)

Để đảm bảo việc đánh giá 5-fold hoàn toàn đồng nhất với các thí nghiệm khác của đội, file `data/processed/folds.csv` phải được đưa lên Kaggle làm Dataset riêng:

1. Đăng nhập vào [Kaggle](https://www.kaggle.com).
2. Vào mục **Datasets** -> Click chọn **New Dataset**.
3. Kéo thả file `data/processed/folds.csv` từ thư mục dự án cục bộ lên.
4. Đặt tên dataset là: `rmit-folds`.
5. Bấm **Create**. Kaggle sẽ tạo dataset tại đường dẫn `/kaggle/input/rmit-folds/folds.csv`.

---

## 3. Thiết Lập Notebook Trên Kaggle (5 Phút)

### Bước 1: Import Notebook
1. Vào mục **Code** trên Kaggle -> Chọn **New Notebook**.
2. Trên thanh menu của Notebook: Chọn **File** -> **Import Notebook**.
3. Tải lên tệp: `notebooks/rembert_5fold_kaggle.ipynb`.
4. Đổi tiêu đề Notebook thành: `RMIT_SGS_BakhiBachKhoa_C1`.

### Bước 2: Thêm Dữ Liệu Đầu Vào (Add Input)
Click vào nút **Add Input** (ở thanh bên phải) và thêm đủ **3 nguồn dữ liệu** sau:
1. **Competition Data:** Tìm `rmit-hackathon-2026` và bấm nút **Add** (chứa `train.csv` và `test.csv`).
2. **Folds Dataset:** Vào tab **Your Work** / **Datasets** -> Chọn dataset `rmit-folds` vừa tạo ở mục 2 và bấm **Add**.
3. **Pretrained Backbone:** Vào tab **Models** -> Tìm kiếm `aisuko/rembert` -> Chọn framework `Transformers`, variation `default` -> Bấm **Add Model**.

### Bước 3: Cấu Hình Session (Settings)
Tại thanh cấu hình bên phải của Notebook (mục **Notebook options**):
- **Accelerator:** Chọn **GPU T4 x2**.
- **Internet:** Chuyển sang chế độ **Off** (BẮT BUỘC theo luật BTC để đảm bảo mô hình chạy hoàn toàn offline).
- **Environment:** Keep preference default (Python 3).
- **Persistence:** Chọn **Files** (hoặc **Variables and Files**).

---

## 4. Chạy Thử Nghiệm Kiểm Tra (Smoke Test) (10 Phút)

Trước khi commit chạy qua đêm, bắt buộc phải chạy Smoke Test trong phiên tương tác để kiểm tra VRAM và đo tốc độ:

1. Kiểm tra Cell 1: Đảm bảo biến `SMOKE_TEST = True`.
2. Bấm **Run All** (hoặc chạy lần lượt từ Cell 0 đến Cell 3).
3. Quá trình kiểm tra sẽ chạy 30 optimizer steps trên Fold 0 với 256 mẫu dữ liệu.
4. **Kiểm tra báo cáo ở Cell 3:**
   - **Peak VRAM:** Phải `< 14.0 GB` (T4 có 16GB VRAM, mức này hoàn toàn an toàn, không bị OOM).
   - **NaN count (`n_bad_steps`):** Phải bằng `0`. Nếu có step bị NaN, cần xem mục Xử Lý Sự Cố bên dưới.
   - **ETA dự kiến:** Xem ước lượng thời gian chạy cho 3 fold trên GPU 0 (thông thường dao động trong khoảng ~3.0 đến 4.5 giờ).

---

## 5. Huấn Luyện Chính Thức Qua Đêm (Save & Run All)

Sau khi Smoke Test thành công:

1. Di chuyển lên **Cell 1**, sửa giá trị:
   ```python
   SMOKE_TEST = False
   ```
2. Kiểm tra chế độ nộp `SUBMIT_MODE`:
   - Mặc định là `"fold_avg"`: Lấy trung bình xác suất của 5 fold checkpoint của RemBERT.
   - Nếu muốn chọn 1 fold duy nhất có điểm validation cao nhất, đặt `SUBMIT_MODE = "best_fold"`.
3. Bấm nút **Save Version** ở góc trên cùng bên phải màn hình:
   - Chọn **Version Type:** `Save & Run All (Commit)`.
   - Bấm **Save**.
4. Bạn có thể an tâm tắt tab trình duyệt và đi ngủ. Kaggle sẽ chạy nền trên 2 GPU T4 song song:
   - GPU 0: Chạy các fold `0`, `2`, `4`.
   - GPU 1: Chạy các fold `1`, `3`.
   - Tổng thời gian hoàn thành dự kiến: **~3.5 – 5 giờ** (giới hạn tối đa của phiên Kaggle là 12 giờ).

---

## 6. Thu Thập Kết Quả Vào Sáng Hôm Sau

1. Mở lại notebook trên Kaggle -> Chuyển sang tab **Output** của version vừa chạy xong.
2. Tải về máy tính các tệp sau:
   - `submission.csv` (và `submission_rembert.csv`): Dự đoán nhãn cho 6,000 dòng test.
   - `metrics.json`: Toàn bộ chỉ số hiệu năng OOF chi tiết.
   - `oof_probs_rembert.npy` & `test_probs_rembert.npy`: Xác suất dự đoán phục vụ phân tích lỗi.
   - Thư mục `logs/`: Ghi log chi tiết của từng fold.
3. Đặt các file tải về vào thư mục của dự án cục bộ: `submissions/rembert/`.

---

## 7. Cách Đọc `metrics.json` Để Chọn Champion Model

Mở tệp `metrics.json` để kiểm tra các chỉ số then chốt:

```json
{
  "model_name": "google/rembert",
  "submit_mode": "fold_avg",
  "oof_macro_f1_base": 0.9412,
  "oof_macro_f1_tuned": 0.9485,
  "multipliers": [1.0, 1.08, 1.25],
  "per_class_f1_tuned": {
    "BENIGN": 0.9780,
    "JAILBREAK": 0.9450,
    "PROMPT_INJECTION": 0.9225
  },
  "per_language_f1": {
    "pt": 0.9510,
    "fr": 0.9465,
    "it": 0.9490,
    "en": 0.9475
  },
  "per_fold": { ... }
}
```

### Các tiêu chí đánh giá model vô địch:
1. **OOF Macro-F1 sau tinh chỉnh (`oof_macro_f1_tuned`):**
   - So sánh trực tiếp với Baseline: `0.9332` (TF-IDF + LinearSVC/LightGBM). Điểm OOF phải cao hơn baseline này.
   - So sánh với các ứng viên single-model khác của team (ví dụ: DeBERTa, XLM-RoBERTa, mmBERT). Model nào có OOF Macro-F1 cao nhất trên cùng 5-fold split này sẽ là ứng viên số 1.
2. **F1 lớp thiểu số `PROMPT_INJECTION`:**
   - Dù chỉ chiếm 7% tập train nhưng `PROMPT_INJECTION` chiếm tới 33.3% tổng điểm metric Macro-F1.
   - Xem giá trị `per_class_f1_tuned["PROMPT_INJECTION"]`. Nếu đạt `>= 0.90`, RemBERT xử lý bài toán instruction injection cực kỳ ấn tượng.
3. **Độ ổn định giữa các fold (`per_fold`):**
   - Kiểm tra `best_f1` của 5 fold trong `per_fold`. Nếu độ chênh lệch giữa fold cao nhất và fold thấp nhất `<= 0.03`, mô hình học tổng quát tốt, không bị overfit cục bộ.
4. **Phân phối nhãn kiểm tra ở Cell 6:**
   - Tỷ lệ dự đoán nhãn `PROMPT_INJECTION` trên tập test (6,000 dòng) cần nằm trong khoảng `5% – 10%`.

---

## 8. Bảng Xử Lý Sự Cố (Troubleshooting)

| Triệu chứng | Nguyên nhân có thể | Cách khắc phục |
|---|---|---|
| `CUDA out of memory` | Bộ nhớ GPU vượt quá 16GB | Vào Cell 1 đặt `BATCH_SIZE = 4`, `GRAD_ACCUM = 4` (effective batch size vẫn là 16). |
| Loss là `NaN` hoặc `n_bad_steps` cao | Learning rate quá lớn hoặc FP16 underflow | Giảm `LR = 1e-5`, đặt `LLRD = 1.0` (tắt LLRD), tăng `WARMUP_RATIO = 0.15`. |
| F1 lớp `PROMPT_INJECTION` = 0 | Mô hình chưa kịp học nhận diện lớp thiểu số | Ở epoch 1 đây là hiện tượng bình thường do warmup. Nếu sang epoch 2 vẫn bằng 0, hãy giảm `LR` xuống `1e-5`. |
| Không tìm thấy model RemBERT | Chưa gắn model input hoặc sai đường dẫn | Kiểm tra lại Bước 3: Đã Add Model `aisuko/rembert` chưa. Script sẽ tự tìm `config.json` có chứa chữ `rembert`. |
| Lỗi không khớp `folds.csv` | Chưa Add Dataset `rmit-folds` hoặc file rỗng | Kiểm tra lại Bước 2 & 3. Script kiểm tra `validate="1:1"`, tuyệt đối không tự sinh fold ngẫu nhiên để bảo vệ tính công bằng khi so sánh. |
| Session bị ngắt / Timeout giữa chừng | Hết thời gian phiên hoặc lỗi mạng | Bật chức năng Resume: Add Output của phiên trước làm Input mới, truyền `--resume_dir /kaggle/input/<tên-output-cũ>`. Các fold đã hoàn thành sẽ tự động được bỏ qua. |
| Dự kiến thời gian chạy quá dài (> 10h) | Tốc độ tính toán bị chậm | Điều chỉnh ở Cell 1: Giảm `EPOCHS = 2`, hoặc chạy ưu tiên 2 fold đại diện (ví dụ fold 0 và 1). |

---

## 9. Quy Trình Trình Nộp Kaggle

Sau khi RemBERT hoàn thành và được xác định là Champion Model của đội:
1. Gửi file `submission.csv` cùng bản tóm tắt `metrics.json` cho **Nhật Tân** (Gatekeeper phụ trách nộp bài của team).
2. Nhật Tân sẽ kiểm tra tính hợp lệ bằng hàm `validate_submission()` trong `src/utils.py` và đối chiếu checklist tuân thủ single-model trước khi chính thức nộp lên bảng xếp hạng Kaggle.
