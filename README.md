# 🚀 RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification

## 🤖 Critical Instructions for AI Coding Agents & Team Members

Applies to Antigravity, Cursor, GitHub Copilot, Claude Code, Windsurf, and all team members.
Read [`AGENTS.md`](file:///d:/Code/rmit-hackathon-2026/AGENTS.md) first before generating or modifying any code.

1. **Bối cảnh đề thi chính thức (Official Ground Truth)**
   - **Cuộc thi:** `rmit-hackathon-2026` — **Challenge 1: Multilingual
     Prompt-Attack Classification** trên Kaggle.
   - **Bản chất bài toán:** Phân loại 3 lớp (**3-Class Classification** trên
     cột `label`):
     - `BENIGN` (~62% tập train): Yêu cầu bình thường, an toàn.
     - `JAILBREAK` (~31% tập train): Thao túng vượt rào an toàn (DAN,
       roleplay, giả định, mã hóa).
     - `PROMPT_INJECTION` (~7% tập train — **lớp thiểu số quyết định**):
       Chiếm quyền điều khiển chỉ thị hệ thống.
   - **Ngôn ngữ thực tế:** **Tiếng Bồ Đào Nha (`pt`), tiếng Pháp (`fr`),
     tiếng Ý (`it`) và tiếng Anh (`en`)**; có sẵn cột `language`. Độ dài văn
     bản có thể lên tới **~18,000 ký tự**.
   - **Metric đánh giá:** **Macro-averaged F1 Score**
     ($\frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} +
     F1_{\text{PROMPT\_INJECTION}})$).
   - **Cấu trúc dữ liệu:**
     - `train.csv` (14,000 dòng): `id`, `text`, `language`, `label`
     - `test.csv` (6,000 dòng): `id`, `text`, `language`
     - `sample_submission.csv` (6,000 dòng): `id`, `label` (giá trị chuỗi:
       `BENIGN`, `JAILBREAK`, hoặc `PROMPT_INJECTION`).
2. **Ràng buộc sống còn (Strict Competition Constraints)**
   - 🚨 **Giới hạn nộp bài:** Tối đa **6 lượt nộp (6 attempts total)** cho
     toàn bộ cuộc thi. Mọi quyết định phải dựa trên điểm 5-Fold OOF Macro-F1;
     chỉ **Nhật Tân** được phép duyệt và nộp bài.
   - 🚨 **Approved Models Only:** Bắt buộc xây dựng giải pháp trên **1 trong
     10 mô hình được BTC phê duyệt** trên Kaggle Models. Không dùng pretrained
     backbone nào khác và ghi rõ tên model ở đầu notebook.

     | Mô hình (Model Name) | Loại | Kaggle Model | Gợi ý / Khuyến nghị |
     | --- | --- | --- | --- |
     | `distilbert/distilbert-base-multilingual-cased` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/Transformers/default) | Baseline nhanh nhất |
     | `jhu-clsp/mmBERT-small` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/mmbert-small/Transformers/default) | Nhanh, đa ngôn ngữ tốt |
     | `jhu-clsp/mmBERT-base` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/mmbert-base/Transformers/default) | Đa ngôn ngữ mạnh mẽ |
     | `FacebookAI/xlm-roberta-base` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/xlm-roberta-base/Transformers/default) | Đa ngôn ngữ rất mạnh |
     | `FacebookAI/xlm-roberta-large` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/xlm-roberta-large/Transformers/default) | Encoder tốt nhất, cần nhiều VRAM |
     | `google/rembert` | Encoder | [Kaggle](https://www.kaggle.com/models/aisuko/rembert/Transformers/default) | Đa ngôn ngữ; khuyến nghị batch size 8 |
     | `protectai/deberta-v3-base-prompt-injection-v2` | Encoder (Safety) | [Kaggle](https://www.kaggle.com/models/aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default) | Đã học sẵn prompt injection |
     | `patronus-studio/wolf-defender-prompt-injection` | Encoder (Safety) | [Kaggle](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default) | Đã học sẵn prompt injection |
     | `ibm-granite/granite-3.3-2b-instruct` | LLM | [Kaggle](https://www.kaggle.com/models/aisuko/granite-3.3-2b-instruct/Transformers/default) | LLM fine-tuning / LoRA |
     | `mistralai/Mistral-7B-Instruct-v0.3` | LLM | [Kaggle](https://www.kaggle.com/models/aisuko/mistral-7b-instruct-v0.3/Transformers/default) | LLM fine-tuning / LoRA |
   - 🚨 **Quy tắc đặt tên notebook:** `<university>_<campus>_<team>_C1`
     (ví dụ: `RMIT_SGS_<team>_C1`).
3. **Quy tắc bắt buộc (Strict Warnings)**
   - ❌ **Không sửa hoặc ghi đè `data/raw/`:** Chỉ đọc (`read-only`). Lưu dữ
     liệu đã xử lý và `folds.csv` vào `data/processed/`.
   - ❌ **Không xóa dấu câu hoặc ký tự Unicode của các ngôn ngữ Romance
     (`pt`, `fr`, `it`) một cách mù quáng:** Không dùng regex
     `[^a-zA-Z0-9]`. Luôn dùng `src/data_loader.py:clean_text_safety()`.
   - ❌ **Không hardcode API keys hoặc Kaggle tokens:** Đọc qua `os.getenv()`
     từ `.env` hoặc `kaggle_secrets`.
   - ❌ **Không sửa notebook (`.ipynb`) của thành viên khác:** Mỗi thành viên
     làm việc trên notebook riêng (`notebooks/01_<ten>_<task>.ipynb`).
   - ✅ **Luôn kiểm tra file nộp bằng `validate_submission()`:** Đảm bảo đúng
     tên cột (`id`, `label`), đúng 6,000 dòng khớp `test.csv`, không có `NaN`,
     và nhãn thuộc `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`.

---

## 👥 1. Phân công Vai trò & Nhiệm vụ 4 Thành viên (Team Roles)

### 1. Nhật Tân — Team Lead, Fast Encoders & Ensemble Master

- Quản lý **quota 6 lần nộp** và kiểm duyệt bài bằng `validate_submission()`.
- Quản lý bộ chia 5-Fold dùng chung tại `data/processed/folds.csv` và Baseline
  3-class (`OOF Macro-F1 = 0.9332`).
- Chạy nhánh Encoder tốc độ cao thuộc 10 Approved Models
  (`jhu-clsp/mmBERT-base`, `jhu-clsp/mmBERT-small`,
  `distilbert/distilbert-base-multilingual-cased`) để chia tải GPU với Trung
  Quân.
- Phân tích lỗi OOF theo từng ngôn ngữ (tập trung vào `BENIGN` vs. `JAILBREAK`
  trên `pt` và `BENIGN` vs. `PROMPT_INJECTION` trên `en, it, fr`).
- Kết hợp ma trận xác suất $(N \times 3)$ và tối ưu hệ số lớp theo từng ngôn
  ngữ bằng `optimize_per_language_multipliers()` để tối đa hóa **Macro-F1**.

### 2. Trung Quân — Deep Learning & Heavy Approved HF Models (Kaggle GPU)

- Fine-tune **Approved Models** chủ lực trên Kaggle GPU:
  `protectai/deberta-v3-base-prompt-injection-v2`,
  `patronus-studio/wolf-defender-prompt-injection`,
  `FacebookAI/xlm-roberta-large`, `FacebookAI/xlm-roberta-base`,
  `google/rembert` (batch size 8), hoặc LLM LoRA (`granite-3.3-2b`,
  `Mistral-7B`).
- Khai báo rõ tên model ở đầu notebook theo quy định của BTC và đặt tên
  `<university>_<campus>_<team>_C1`.
- Xử lý mất cân bằng 62–31–7 bằng **Class Weights / Focal Loss**, tận dụng
  Self-Attention để giải quyết lỗi *Contrastive Benign vs. Subtle Jailbreak*
  trong tiếng Bồ Đào Nha (`pt`).
- Áp dụng **Head+Tail Truncation** cho văn bản dài (tới 18k ký tự) và gắn tag
  `language`; xuất xác suất OOF `(14000, 3)` và Test `(6000, 3)`.

### 3. Cường Thịnh — Multilingual Security & NLP Feature Engineering

- Giải quyết điểm nghẽn #1 (chiếm 80% tổng lỗi): Phân biệt **Contrastive
  Benign** (mượn văn phong nhập vai hỏi việc vô hại) với **Subtle Jailbreak**
  (đóng vai học thuật hỏi việc độc hại) trong tiếng Bồ Đào Nha (`pt` — chiếm
  45% dữ liệu).
- Giải quyết điểm nghẽn #2 (chiếm 18% tổng lỗi): Bắt lệnh `PROMPT_INJECTION`
  ngắn gọn trong `en, it, fr`.
- Khai thác **Quy luật Độ Dài Vàng** (`char_len < 112` $\rightarrow$ 0%
  `JAILBREAK`; `char_len > 3752` $\rightarrow$ 0% `JAILBREAK`;
  `char_len > 5128` $\rightarrow$ 100% `BENIGN`) và bàn giao DataFrame đặc
  trưng số/boolean cho Khánh Linh.

### 4. Khánh Linh — Tabular ML, Per-Language Models & Stacking Specialist

- Huấn luyện mô hình 3 lớp trên CPU local: `LightGBM`
  (`objective="multiclass"`, `num_class=3`, `class_weight="balanced"`),
  `CatBoost`, `LogisticRegression`, và `LinearSVC`.
- Kết hợp TF-IDF đa ngôn ngữ (Word `(1, 3)` + Char `char_wb` `(2, 5)`),
  `TruncatedSVD`, features của Cường Thịnh, và xác suất OOF từ các mô hình
  tuyến tính làm meta-features (Stacking).
- Xây dựng thêm **Per-Language Specialist Models** (đặc biệt là model chuyên
  biệt cho `pt` — chiếm 45% dữ liệu).
- Đánh giá 5-Fold CV (`folds.csv`) theo **Macro-F1**; xuất xác suất OOF/Test
  `(N, 3)` và file nộp `id, label`.

---

## 📅 2. Thông tin nhanh & Lịch trình sống còn (Day 1 & Day 2)

- **Địa điểm:** RMIT Saigon South Campus, 702 Nguyễn Văn Linh, Tân Hưng,
  Q.7, TP.HCM.
- **Lịch trình Day 1 (Thứ Năm, 22/10/2026):**
  - `08:15–09:00`: **Check-in theo cả team tại Sảnh Building 2**
    *(Cả đội phải có mặt cùng lúc trước 9:00 AM, trễ sẽ bị loại hoặc trừ 10%
    tổng điểm.)*
  - `09:30 – 10:00`: Briefing & rà soát quy chế.
  - `10:00–12:00`: Challenge Session 1 — Chốt điểm CV 5-Fold cho Baseline
    3-class, LightGBM đa ngôn ngữ và các fold đầu của DeBERTa/XLM-R; thực
    hiện lượt nộp mở hàng có kiểm soát (Submission #1/6).
  - `12:00 – 13:00`: Nghỉ ăn trưa.
  - `13:00–16:30`: Challenge Session 2 — Tinh chỉnh Feature Engineering
    (`pt, fr, it, en`), hoàn tất OOF `(14000, 3)` từ các Approved Models,
    chạy Blending và tối ưu ngưỡng lớp `optimize_class_multipliers` cho
    Macro-F1.
  - `16:30–18:00`: **CODE FREEZE (Đóng băng tính năng mới)** — Kiểm định kỹ
    các lượt nộp cuối (tối đa 6 lượt toàn giải), chọn Final Submissions có CV
    + LB ổn định nhất, hoàn thiện tên notebook `RMIT_SGS_<team>_C1`.
  - `18:00`: **HẠN CHÓT NỘP BÀI (Submission Deadline).**
- **Lịch trình Day 2 (Thứ Sáu, 23/10/2026):**
  - `10:00–13:30`: Check-in, Lễ trao giải liên cơ sở (Melbourne Theatre,
    Building 1) và Networking.

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
