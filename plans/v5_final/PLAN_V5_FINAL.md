# 🎯 PLAN FINAL — LƯỢT NỘP 6/6 (RemBERT V5) — Deadline 18:00 ICT (22/10/2026)

> Team `BakhiBachKhoa` · Notebook `RMIT_SGS_BakhiBachKhoa_C1` · Model duy nhất: `google/rembert` (`aisuko/rembert`)
> Plan chốt lúc 15:40 ICT. **Notebook V5 đã được dựng sẵn và kiểm định offline** → việc còn lại chỉ là upload, commit, giám sát và nộp.

---

## 0. Quyết định đã chốt (KHÔNG thảo luận lại)

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Số fold | **4 fold (0,1,2,3)**, GPU0: 0→2, GPU1: 1→3 | 5 fold (3+2) cần ~130–140 phút → không kịp. 2+2 ≈ 85–90 phút |
| Epochs | **3** (cosine kết thúc ở epoch 3) | Log V2/V3: cuối epoch 3 ≈ đỉnh (0.9619 vs 0.9627) |
| Eval/Snapshot | **1 lần duy nhất ở cuối**, `TOP_K_CHECKPOINTS=1` | Top-2 snapshot chỉ ±0.0004, tốn ~7 phút/fold |
| Loss | `SampleWeightedLabelSmoothingLoss` (V3), smoothing 0.03, **weights `[1.0, 1.8, 7.0]`** | V3 thật ra chạy `[1,1.5,5.0]` → hụt en-PI (64/84 so với 75/84 của V2 trên fold 0–3). 7.0 nằm giữa 5.0 và 8.8 |
| LR / LLRD | **1.5e-5 / 0.965** (giữ nguyên) | Đây là cấu hình THẬT của cả V2 lẫn V3 (theo `config.json`; HANDOFF ghi sai) |
| Fold | Tự sinh từ `train.csv` với key **`.lower()`** | Đã kiểm chứng local: **khớp 100% `data/processed/folds.csv`** (14,000/14,000). Không cần mount `rmit-folds` |
| Hậu xử lý | 1 hệ số J toàn cục + 1 hệ số PI/ngôn ngữ, **kẹp [0.70, 1.40]**, co 50% về 1.0, chỉ dùng khi Nested-CV tăng ≥ 0.0005 | Không thể tái diễn lỗi `it × 0.3` của V2 |
| Chốt chặn thời gian | `DEADLINE_UTC = 10:40 UTC` (= **17:40 ICT**) → kill process, ghép submission từ fold đã xong | Bảo đảm notebook luôn kết thúc và có `submission.csv` |
| Batch / VRAM | Giữ `BATCH_SIZE=8, GRAD_ACCUM=2`, gradient checkpointing, MAX_LEN 512 | Code V3 đã chạy ổn (10.7 GB). Không thay đổi để tránh OOM |

**Tuân thủ luật:** 1 backbone duy nhất (`google/rembert`), input chỉ gồm data đề + `aisuko/rembert`, không nạp `.npy` cũ, `submission.csv` sinh trực tiếp từ output của 1 lần commit.

---

## 1. File liên quan

| File | Vai trò |
|---|---|
| [`notebooks/rembert_v5_kaggle.ipynb`](file:///d:/Code/rmit-hackathon-2026/notebooks/rembert_v5_kaggle.ipynb) | **Notebook nộp bài (đã dựng sẵn)**: upload thẳng lên Kaggle |
| [`plans/v5_final/build_v5_notebook.py`](file:///d:/Code/rmit-hackathon-2026/plans/v5_final/build_v5_notebook.py) | Script dựng V5 từ V3 (diff tối thiểu, có assert và kiểm tra cú pháp) |
| [`plans/v5_final/cell4_launch_v5.py`](file:///d:/Code/rmit-hackathon-2026/plans/v5_final/cell4_launch_v5.py) | Cell 4 mới: launch 2 GPU + hard deadline |
| [`plans/v5_final/cell5_aggregate_v5.py`](file:///d:/Code/rmit-hackathon-2026/plans/v5_final/cell5_aggregate_v5.py) | Cell 5 mới: ghép fold đã xong, hậu xử lý kẹp biên, sinh submission (đã test local trên npy V3: chạy 10 s) |
| [`plans/v5_final/MASTER_PROMPT_ARGON.md`](file:///d:/Code/rmit-hackathon-2026/plans/v5_final/MASTER_PROMPT_ARGON.md) | Prompt giao cho Argon |

Diff so với V3 (toàn bộ, không có gì khác):
1. Cell 0: tiêu đề V5 (dòng 1 vẫn là `# Model: google/rembert`).
2. Cell 1: `VERSION_TAG="V5"`, `EPOCHS=3`, `CLASS_WEIGHTS=[1.0,1.8,7.0]`, `TOP_K_CHECKPOINTS=1`, `GPU_FOLDS={"0":[0,2],"1":[1,3]}`.
3. Cell 2 (`train_rembert.py`): key gom nhóm fold thêm `.lower()`; `eval_schedule = {(epochs-1, len(train_loader)-1)}`.
4. Cell 4: thay bằng `cell4_launch_v5.py`.
5. Cell 5: thay bằng `cell5_aggregate_v5.py`.

---

## 2. Timeline thực thi

| Giờ (ICT) | Việc | Ai |
|---|---|---|
| 15:40–15:55 | Upload `rembert_v5_kaggle.ipynb` vào notebook `RMIT_SGS_BakhiBachKhoa_C1` (version mới). Input: competition data + model `aisuko/rembert`. Accelerator **GPU T4 x2**. Internet tắt cũng được | Argon + User |
| **≤ 16:00** | **Save Version → Save & Run All (Commit)**. KHÔNG chạy smoke test riêng (V3 code đã chạy ổn; commit log 6 phút đầu đóng vai smoke test) | User |
| 16:00–16:08 | Theo dõi log commit. Phải thấy ở **cả 2 GPU**: `Tổng mẫu train: 11,200 \| Val: 2,800 \| Steps/epoch: 700 \| Total opt steps: 2100` | Argon |
| ~16:50 | Fold 0 & 1 xong (~42–45 phút/fold): log `[HOÀN THÀNH FOLD x] ... Thời gian: ~4x phút` | Argon |
| ~17:30 | Fold 2 & 3 xong, Cell 5/6 chạy, commit kết thúc | — |
| 17:40 | **Hard deadline trong notebook** (nếu chậm: kill và ghép từ fold đã xong) | tự động |
| 17:40–17:52 | Tải output, chạy Gate, nộp 6/6, chọn 2 bài cho Private | Nhật Tân (Gatekeeper) |
| 18:00 | Đóng cổng | — |

### Phương án dự phòng
- **Commit lỗi trong 8 phút đầu** (path / import / OOM): sửa và commit lại ngay. Nếu commit lại bắt đầu **sau 16:15** thì vẫn chạy: hard deadline 17:40 sẽ cắt ở 2 fold (0,1). Cell 5 tự xử lý.
- **Commit bắt đầu sau 16:50:** KHÔNG chạy nữa. Giữ nguyên 5 lượt hiện tại, chọn **V2 + V3** cho Private.
- **Một GPU crash giữa chừng:** Cell 4 không raise lỗi, Cell 5 ghép từ các fold còn lại (tối thiểu 1 fold).
- **Kaggle lệch giờ / queue lâu:** `DEADLINE_UTC` là mốc tuyệt đối 10:40 UTC. Nếu commit bắt đầu muộn thì tự cắt sớm hơn, không cần sửa.

---

## 3. Gate trước khi nộp (Gatekeeper Nhật Tân)

### 3.1 Kiểm tra bắt buộc (fail là KHÔNG nộp)
1. `submission.csv`: đúng **6000 dòng**, cột `id,label`, không NaN, không trùng id, nhãn thuộc `{BENIGN, JAILBREAK, PROMPT_INJECTION}`. Cell 6 đã assert sẵn.
2. Tỷ lệ PI trên test nằm trong **5%–10%** (V3 ≈ 6.8%).
3. `metrics.json → folds_done` có ít nhất 2 fold.
4. Chạy local: `python -c "import pandas as pd; from src.utils import validate_submission; validate_submission(pd.read_csv('submissions/v5_rembert.csv'))"`.

### 3.2 Bảng tham chiếu OOF trên CÙNG các hàng (cùng `folds.csv`)

| Tập hàng | V2 argmax | V3 argmax | en-PI bắt trúng V2 / V3 |
|---|---|---|---|
| Fold {0,1} (5,600 hàng) | 0.96381 (166 lỗi) | 0.96571 (153 lỗi) | 27/32 · 24/32 |
| Fold {0,1,2,3} (11,200 hàng) | **0.96473** (312 lỗi) | **0.96498** (300 lỗi) | **75/84 · 64/84** |

> Lưu ý: V3 dùng bộ fold khác (không `.lower()`), nên số V3 ở trên là OOF của V3 trên cùng các hàng chứ không cùng phép chia. V2 thì so được trực tiếp.

### 3.3 Quy tắc nộp và chọn bài (Private = 70%)
- **Nộp lượt 6/6 = V5** nếu qua 3.1 (đây là quyền chọn miễn phí: nộp rồi mà không chọn thì không mất gì).
- **Chọn 2 bài cho Private:**
  - **Luôn chọn V2** (`0.97471`, PB tốt nhất).
  - Bài thứ 2 là **V5** nếu `oof_macro_f1_base` (4 fold) **≥ 0.9650** VÀ Public LB của V5 **≥ 0.9720**. Ngược lại chọn **V3** (`0.97245`).
  - Nếu V5 chỉ xong 2 fold: dùng ngưỡng OOF **≥ 0.9655** trên fold {0,1}.
- Ghi vào submission log: tên file `submissions/v5_rembert.csv`, OOF, rule hậu xử lý, Public LB, người nộp.

---

## 4. Rủi ro đã biết

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| 3 epoch kém hơn 4 epoch một chút | Thấp | Cosine được scale lại theo 3 epoch; log V2/V3 cho thấy cuối epoch 3 ≈ đỉnh |
| 4 fold thay vì 5 (mỗi model học 80% data, ensemble 4 thay vì 5) | Thấp | Chênh lệch kỳ vọng < 0.001 |
| Weight 7.0 tăng báo động giả PI ở pt | Trung bình | Hậu xử lý kẹp biên có hệ số PI/ngôn ngữ (cho phép giảm về 0.85 sau khi co) |
| Ước tính thời gian sai (> 50 phút/fold) | Trung bình | Hard deadline 17:40 → vẫn có submission từ 2 fold |
| Kaggle queue GPU lâu | Trung bình | Commit ≤ 16:00. Nếu > 16:50 thì hủy, chọn V2 + V3 |
