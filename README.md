# 🚀 RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification

## 🤖 Critical Instructions for AI Coding Agents & Team Members

Applies to Antigravity, Cursor, GitHub Copilot, Claude Code, Windsurf, and all team members.
Read [`AGENTS.md`](file:///d:/Code/rmit-hackathon-2026/AGENTS.md) first before generating or modifying any code.

1. **Bối cảnh đề thi chính thức (Official Ground Truth):**
   - **Cuộc thi:** `rmit-hackathon-2026` — **Challenge 1: Multilingual Prompt-Attack Classification** trên Kaggle.
   - **Bản chất bài toán:** Phân loại 3 lớp (**3-Class Classification** trên cột `label`):
     - `BENIGN` (~62% tập train): Yêu cầu bình thường, an toàn.
     - `JAILBREAK` (~31% tập train): Thao túng vượt rào an toàn (DAN, roleplay, giả định, mã hóa).
     - `PROMPT_INJECTION` (~7% tập train — **lớp thiểu số quyết định**): Chiếm quyền điều khiển chỉ thị hệ thống.
   - **Ngôn ngữ thực tế:** **Tiếng Bồ Đào Nha (`pt`), tiếng Pháp (`fr`), tiếng Ý (`it`) và tiếng Anh (`en`)**, có sẵn cột `language`. Độ dài văn bản có thể lên tới **~18,000 ký tự**.
   - **Metric đánh giá:** **Macro-averaged F1 Score** ($\frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} + F1_{\text{PROMPT\_INJECTION}})$).
   - **Cấu trúc dữ liệu:**
     - `train.csv` (14,000 dòng): `id, text, language, label`
     - `test.csv` (6,000 dòng): `id, text, language`
     - `sample_submission.csv` (6,000 dòng): `id, label` (Giá trị chuỗi: `BENIGN`, `JAILBREAK`, hoặc `PROMPT_INJECTION`).
2. **Ràng buộc sống còn (Strict Competition Constraints):**
   - 🚨 **Giới hạn nộp bài:** **Tối đa chỉ có 6 lượt nộp (6 attempts total)** cho toàn bộ cuộc thi. Mọi quyết định phải dựa trên điểm 5-Fold OOF Macro-F1; chỉ **Nhật Tân** được phép duyệt và nộp bài.
   - 🚨 **Approved Models Only:** Bắt buộc sử dụng các backbone trong danh sách được BTC phê duyệt trên Kaggle Models:
     - `protectai/deberta-v3-base-prompt-injection-v2`
     - `FacebookAI/xlm-roberta-base`
     - `patronus-studio/wolf-defender-prompt-injection`
     - `distilbert-base-multilingual-cased`
   - 🚨 **Quy tắc đặt tên Notebook nộp:** `<university>_<campus>_<team>_C1` (Ví dụ: `RMIT_SGS_<team>_C1`).
3. **QUY TẮC CẤM VI PHẠM (STRICT WARNINGS):**
   - ❌ **KHÔNG BAO GIỜ sửa đổi hoặc ghi đè dữ liệu trong `data/raw/`:** Chỉ đọc (`read-only`). Dữ liệu đã xử lý hoặc `folds.csv` lưu vào `data/processed/`.
   - ❌ **KHÔNG BAO GIỜ xóa dấu câu hoặc ký tự Unicode ngôn ngữ Romance (`pt, fr, it`) một cách mù quáng:** Tuyệt đối không dùng regex `[^a-zA-Z0-9]`. Luôn dùng `src/data_loader.py:clean_text_safety()`.
   - ❌ **KHÔNG BAO GIỜ hardcode API Keys hoặc Kaggle Token vào code:** Luôn đọc qua `os.getenv()` từ `.env` hoặc `kaggle_secrets`.
   - ❌ **KHÔNG sửa chung một file Jupyter Notebook (`.ipynb`) của thành viên khác:** Mỗi thành viên làm việc trên notebook riêng (`notebooks/01_<ten>_<task>.ipynb`).
   - ✅ **LUÔN kiểm tra định dạng `submission.csv` bằng `validate_submission()`:** Đảm bảo đúng tên cột (`id`, `label`), đúng `6,000` dòng khớp `test.csv`, không có `NaN`, và nhãn thuộc đúng tập `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`.

---

## 👥 1. Phân công Vai trò & Nhiệm vụ 4 Thành viên (Team Roles)

| Thành viên | Vai trò chính | Nhiệm vụ cốt lõi & Đầu ra (Deliverables) |
| :--- | :--- | :--- |
| **1. Nhật Tân** | **Team Lead, Kaggle Pipeline & Ensemble Master** | • Quản lý nghiêm ngặt **quota 6 lần nộp** toàn giải & kiểm duyệt qua `validate_submission()`.<br>• Sinh file chia 5-Fold chuẩn `data/processed/folds.csv` cho cả đội.<br>• Xây dựng Baseline 3-class (`TF-IDF + language + LogisticRegression / LinearSVC`).<br>• Phân tích lỗi nhầm lẫn giữa `JAILBREAK` và `PROMPT_INJECTION` trên OOF.<br>• Blending ma trận xác suất $(N \times 3)$ & tối ưu ngưỡng/trọng số lớp (`optimize_class_multipliers`) để cực đại hóa **Macro-F1**. |
| **2. Trung Quân** | **Deep Learning & Approved HF Models (Kaggle GPU)** | • Fine-tune các **Approved Models** (`protectai/deberta-v3-base-prompt-injection-v2`, `FacebookAI/xlm-roberta-base`, `patronus-studio/wolf-defender-prompt-injection`, `distilbert-base-multilingual-cased`) trên Kaggle GPU.<br>• Xử lý mất cân bằng 62-31-7 bằng **Class Weights / Focal Loss**.<br>• Áp dụng **Head+Tail Truncation** cho văn bản dài (tới 18k ký tự) & gắn tag `language`.<br>• Quản lý notebook `<university>_<campus>_<team>_C1` & xuất xác suất OOF `(14000, 3)` + Test `(6000, 3)`. |
| **3. Cường Thịnh** | **Multilingual Security & NLP Feature Engineering** | • Bóc tách đặc trưng tấn công Jailbreak vs Prompt Injection trên 4 ngôn ngữ **Pháp (`fr`), Ý (`it`), Bồ Đào Nha (`pt`), Anh (`en`)**.<br>• Xây dựng bộ regex phân biệt lệnh chiếm quyền chỉ thị (`PROMPT_INJECTION`) với nhập vai vượt rào (`JAILBREAK`).<br>• Khai thác cột `language` và các đặc trưng dị thường của văn bản siêu dài (tới 18k ký tự).<br>• Bàn giao DataFrame đặc trưng số/boolean cho Khánh Linh. |
| **4. Khánh Linh** | **Tabular ML & Multiclass TF-IDF Specialist** | • Huấn luyện mô hình 3 lớp trên CPU local: `LightGBM` (`objective="multiclass"`, `num_class=3`, `class_weight="balanced"`), `CatBoost`, `LogisticRegression`, `LinearSVC`.<br>• Kết hợp TF-IDF đa ngôn ngữ (Word `(1, 3)` + Char `char_wb` `(2, 5)`) + cột `language` + features của Cường Thịnh.<br>• Đánh giá 5-Fold CV (`folds.csv`) theo **Macro-F1**, xuất xác suất OOF/Test `(N, 3)` và file nộp `id, label`. |

---

## 📅 2. Thông tin nhanh & Lịch trình sống còn (Day 1 & Day 2)

- **Địa điểm:** RMIT Saigon South Campus, 702 Nguyễn Văn Linh, Tân Hưng, Q.7, TP.HCM.
- **Lịch trình Day 1 (Thứ Năm, 22/10/2026):**
  - `08:15 – 09:00`: **Check-in theo cả team tại Sảnh Building 2** *(Cả đội phải có mặt cùng lúc trước 9:00 AM, trễ sẽ bị loại hoặc trừ 10% tổng điểm)*.
  - `09:30 – 10:00`: Briefing & rà soát quy chế.
  - `10:00 – 12:00`: Challenge Session 1 — Chốt điểm CV 5-Fold cho Baseline 3-class, LightGBM đa ngôn ngữ và các fold đầu của DeBERTa/XLM-R; thực hiện lượt nộp mở hàng có kiểm soát (Submission #1/6).
  - `12:00 – 13:00`: Nghỉ ăn trưa.
  - `13:00 – 16:30`: Challenge Session 2 — Tinh chỉnh Feature Engineering (`pt, fr, it, en`), hoàn tất OOF `(14000, 3)` từ các Approved Models, chạy Blending & tối ưu hóa ngưỡng lớp `optimize_class_multipliers` cho Macro-F1.
  - `16:30 – 18:00`: **CODE FREEZE (Đóng băng tính năng mới)** — Kiểm định kỹ các lượt nộp cuối (tối đa 6 lượt toàn giải), chọn Final Submissions có CV + LB ổn định nhất, hoàn thiện tên notebook `RMIT_SGS_<team>_C1`.
  - `18:00`: **HẠN CHÓT NỘP BÀI (Submission Deadline).**
- **Lịch trình Day 2 (Thứ Sáu, 23/10/2026):**
  - `10:00 – 13:30`: Check-in, Lễ trao giải liên cơ sở (Melbourne Theatre, Building 1) & Networking.

---

## 📂 3. Cấu trúc thư mục dự án

```text
rmit-hackathon-2026/
├── docs/                      # Tài liệu, booklet RMIT, cheat sheet đa ngôn ngữ (pt/fr/it/en)
├── data/
│   ├── raw/                   # Dữ liệu gốc Kaggle: train.csv, test.csv (TUYỆT ĐỐI KHÔNG SỬA)
│   └── processed/             # folds.csv dùng chung và dữ liệu đã tạo đặc trưng
├── notebooks/                 # Thử nghiệm cá nhân (00_baseline_demo.ipynb, 01_<ten>_<task>.ipynb)
├── src/                       # Code Python dùng chung cho cả team
│   ├── config.py              # Đường dẫn thư mục, hằng số, Random Seed
│   ├── data_loader.py         # Hàm đọc dữ liệu, clean_text_safety, và đặc trưng đa ngôn ngữ
│   ├── folds.py               # Hàm tạo và đọc bộ chia 5-Fold thống nhất (folds.csv)
│   ├── models.py              # Pipeline mô hình 3 lớp (TF-IDF + language + LR/LinearSVC/LGBM)
│   ├── blend.py               # Blending ma trận xác suất (N x 3) & tối ưu ngưỡng Macro-F1
│   └── utils.py               # Hàm tính Macro-F1 (kèm F1 từng lớp) và kiểm tra submission (id, label)
├── submissions/               # Các phiên bản file nộp chuẩn (id, label) lên Kaggle
├── .env                       # API Key cá nhân (Không push lên Git)
├── .env.example               # Mẫu biến môi trường
├── AGENTS.md                  # Chỉ dẫn ngữ cảnh bắt buộc cho AI Coding Agents
├── prompt.md                  # Mẫu prompt phân vai cho 4 thành viên
├── README.md                  # Tài liệu hướng dẫn chung cho cả team
└── requirements.txt           # Các thư viện Python cần cài đặt
```

---

## ⚡ 4. Hướng dẫn thiết lập & Chạy nhanh (Quick Start)

### Cài đặt môi trường

```bash
# Kích hoạt virtual environment hiện có (hoặc tạo mới)
# Windows:
.\.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Chạy kiểm thử Pipeline 3-Class Baseline

```bash
python -m src.models
```

Hoặc mở trực tiếp [`notebooks/00_baseline_demo.ipynb`](file:///d:/Code/rmit-hackathon-2026/notebooks/00_baseline_demo.ipynb).
