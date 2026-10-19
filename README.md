# 🚀 RMIT Hackathon 2026 — Team Repository

**Theme:** `Security x Generative AI x Low-Resource Languages`  
**Repository:** `https://github.com/nhatnews/rmit-hackathon-2026.git`
(PRIVATE)

## 🤖 Critical Instructions for AI Coding Agents

Applies to Antigravity, Cursor, GitHub Copilot, Claude Code, Windsurf, etc.
Read this first before generating or modifying any code.

1. **Bối cảnh bài toán (Context):**
   - Đây là codebase thi đấu **RMIT Hackathon 2026** trên nền tảng **Kaggle**
     với thời gian giới hạn (10:00 AM–6:00 PM ngày 22/10/2026).
   - Trọng tâm bài toán là giao thoa giữa **AI Safety/Security** (Jailbreak
     Detection, Prompt Injection, Red Teaming), **Generative AI (LLMs)** và
     **Ngôn ngữ ít tài nguyên (Low-Resource Languages)** như tiếng Việt,
     tiếng Mã Lai, văn bản trộn ngôn ngữ (code-switching).
2. **Triết lý "Baseline First" (Ưu tiên tốc độ & sự ổn định):**
   - Luôn ưu tiên giải pháp chạy được ngay, đơn giản, ít phụ thuộc (ví dụ:
     `TF-IDF + LogisticRegression / LightGBM` hoặc gọi API gọn nhẹ) trước khi
     đề xuất các mô hình phức tạp.
   - Không tự ý viết các đoạn code huấn luyện quá nặng làm treo máy local nếu
     người dùng chưa yêu cầu.
3. **QUY TẮC CẤM VI PHẠM (STRICT WARNINGS):**
   - ❌ **KHÔNG BAO GIỜ sửa đổi hoặc ghi đè dữ liệu trong `data/raw/`:** Mọi dữ
     liệu gốc tải từ Kaggle chỉ được đọc (`read-only`). Kết quả tiền xử lý
     phải lưu vào `data/processed/`.
   - ❌ **KHÔNG BAO GIỜ xóa dấu câu hoặc ký tự Unicode tiếng Việt/ngôn ngữ bản
     địa một cách mù quáng:** Khi viết hàm `clean_text`, tuyệt đối không dùng
     regex kiểu `[^a-zA-Z0-9]` vì sẽ làm mất toàn bộ dấu tiếng Việt và đặc
     trưng của câu lệnh tấn công (adversarial tokens/obfuscation).
   - ❌ **KHÔNG BAO GIỜ hardcode API Keys hoặc Kaggle Token vào code:** Luôn
     đọc qua `os.getenv()` từ file `.env` (trên Local) hoặc `kaggle_secrets`
     (trên Kaggle Notebook).
   - ❌ **KHÔNG sửa chung một file Jupyter Notebook (`.ipynb`) của thành viên
     khác:** Để tránh lỗi Git Merge Conflict trên file JSON của Notebook, mỗi
     thành viên làm việc trên file `.ipynb` riêng mang tên mình. Các hàm dùng
     chung phải được đóng gói vào `src/*.py`.
   - ✅ **LUÔN kiểm tra định dạng `submission.csv` trước khi xuất:** Đảm bảo
     đúng tên cột (`id`, `target`), đúng số lượng dòng so với `test.csv`,
     không chứa `NaN`/`null`/`inf`, và kiểm tra kỹ xem đề yêu cầu nộp xác suất
     (`predict_proba` cho ROC-AUC) hay nhãn cứng (`0/1` cho F1/Accuracy).

## 📅 1. Thông tin nhanh & Lịch trình sống còn (Day 1 & Day 2)

- **Địa điểm:** RMIT Saigon South Campus, 702 Nguyễn Văn Linh, Tân Hưng,
  Q.7, TP.HCM.
- **Lịch trình Day 1 (Thứ Năm, 22/10/2026):**
  - `08:15 – 09:00`: **Check-in theo cả team tại Sảnh Building 2**
    *(Cả đội phải có mặt cùng lúc trước 9:00 AM, trễ sẽ bị loại hoặc trừ 10%
    tổng điểm)*.
  - `09:30 – 10:00`: Công bố đề thi (Task Release & Briefing).
  - `10:00 – 12:00`: Challenge Session 1 — **Mục tiêu: Có 1 bản nộp Baseline
    thành công trên Kaggle trước 12:00 trưa.**
  - `12:00 – 13:00`: Nghỉ ăn trưa.
  - `13:00 – 16:30`: Challenge Session 2 — Tinh chỉnh mô hình, thử nghiệm
    Prompt / Feature Engineering.
  - `16:30 – 18:00`: **CODE FREEZE (Đóng băng tính năng mới)** — Kiểm tra lại
    file nộp tốt nhất, hoàn thiện báo cáo/slide.
  - `18:00`: **HẠN CHÓT NỘP BÀI (Submission Deadline).**
- **Lịch trình Day 2 (Thứ Sáu, 23/10/2026):**
  - `10:00 – 13:30`: Check-in, Lễ trao giải liên cơ sở (Melbourne Theatre,
    Building 1) & Networking.

## 📂 2. Cấu trúc thư mục dự án

```text
rmit-hackathon-2026/
├── docs/                      # Tài liệu, booklet RMIT, ghi chú đề bài & mẫu prompt
├── data/
│   ├── raw/                   # Dữ liệu gốc Kaggle (TUYỆT ĐỐI KHÔNG SỬA)
│   └── processed/             # Dữ liệu đã qua làm sạch / tạo đặc trưng
├── notebooks/                 # Thử nghiệm cá nhân (01_<ten>_baseline.ipynb)
├── src/                       # Code Python dùng chung cho cả team
│   ├── config.py              # Đường dẫn thư mục, hằng số, Random Seed
│   ├── data_loader.py         # Hàm đọc và làm sạch dữ liệu
│   ├── models.py              # Pipeline mô hình (Baseline TF-IDF, LLM API, v.v.)
│   └── utils.py               # Hàm tính điểm (AUC/F1) và kiểm tra submission
├── submissions/               # Các phiên bản file nộp lên Kaggle
├── .env                       # API Key cá nhân (Không push lên Git)
├── .env.example               # Mẫu biến môi trường
├── AGENTS.md                  # Chỉ dẫn ngữ cảnh cho AI Coding Agents
├── README.md                  # Tài liệu hướng dẫn chung cho cả team
└── requirements.txt           # Các thư viện Python cần cài đặt
```

## ⚡ 3. Hướng dẫn thiết lập & Chạy nhanh (Quick Start)

### Cài đặt môi trường

```bash
# Tạo và kích hoạt virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Cấu hình `.env`

Sao chép `.env.example` thành `.env` và điền Kaggle API credentials / LLM
Keys cá nhân:

```bash
cp .env.example .env
```

### Chạy Baseline mẫu

```bash
python -m src.models
```

Hoặc mở notebook trong thư mục `notebooks/`.
