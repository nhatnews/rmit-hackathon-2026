# 🤖 AGENTS.md — Critical Instructions for AI Coding Agents
*(Applies to Antigravity, Cursor, Copilot, Claude Code, Windsurf, and any automated AI coding assistant)*

## 1. Hackathon Context & Domain
- **Competition:** RMIT Hackathon 2026 (Saigon South Campus) on Kaggle.
- **Theme:** `Security x Generative AI x Low-Resource Languages`
  - AI Safety & Security: Jailbreak Detection, Prompt Injection, Adversarial Obfuscation, Red Teaming.
  - Low-Resource Languages: Vietnamese, Malay, Regional dialects, Code-switching (multilingual mix).
- **Time constraint:** 10:00 AM – 6:00 PM (8-hour sprint). Every minute counts!

---

## 2. Core Philosophy: "Baseline First" (Speed & Stability)
1. **Ship a valid submission early:** Target having a valid submission file submitted to Kaggle before 12:00 PM.
2. **Start simple:** Always implement and benchmark standard baselines first:
   - Feature Extraction: `TfidfVectorizer` (word-level + char n-grams `(2, 5)` to capture adversarial token splitting and typo obfuscation).
   - Classifiers: `LogisticRegression`, `LinearSVC`, or `LightGBM`.
   - Lightweight API-based evaluation when appropriate.
3. **Never freeze the local machine:** Do NOT launch long-running deep learning training jobs locally without explicit user instruction. Suggest Kaggle Notebook GPU (P100 / T4) for heavy neural models (mDeBERTa, XLM-RoBERTa, etc.).

---

## 3. Strict Operating Rules (Zero Exceptions)

| Rule | Enforcement |
| :--- | :--- |
| ❌ **NEVER modify or overwrite `data/raw/`** | `data/raw/` is **strictly READ-ONLY**. All cleaned, tokenized, augmented, or engineered data must be saved under `data/processed/`. |
| ❌ **NEVER strip Vietnamese / Indigenous Unicode blindly** | **NEVER** use generic sanitization regex such as `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. Doing so strips Vietnamese diacritics (`á, ế, ơ, ư, đ...`), Malay text, and crucially strips adversarial punctuation used in prompt injection (e.g., base64 delimiters, markdown escapes, zero-width characters). Use `src/data_loader.py:clean_text_safety()`. |
| ❌ **NEVER hardcode API keys or secrets** | Always retrieve keys via `os.getenv()` from `.env` (locally) or Kaggle Secrets (on Kaggle Notebook). Never commit `.env` or `kaggle.json`. |
| ❌ **NEVER edit another member's `.ipynb` file** | To prevent Git merge conflicts on notebook JSON metadata, each member must maintain their own notebook (e.g. `notebooks/01_<name>_<task>.ipynb`). All reusable logic must be refactored into `src/*.py`. |
| ✅ **ALWAYS validate `submission.csv` before saving** | Use `src/utils.py:validate_submission()`. Verify: column names (`id`, `target`), exact row count matching `test.csv`, no `NaN`/`null`/`inf`, correct output type (probabilities `[0, 1]` for ROC-AUC vs binary labels `0/1` for F1/Accuracy). |

---

## 4. Codebase Architecture
```text
rmit-hackathon-2026/
├── docs/                      # Rules, competition booklet, prompt templates
├── data/
│   ├── raw/                   # Raw Kaggle data (STRICTLY READ-ONLY)
│   └── processed/             # Cleaned / engineered datasets
├── notebooks/                 # Individual experiment notebooks (01_<name>_xxx.ipynb)
├── src/                       # Shared team Python modules
│   ├── config.py              # Directory paths, random seeds, environment helpers
│   ├── data_loader.py         # Safe text loader, Unicode-preserving cleaner
│   ├── models.py              # Baseline pipelines (TF-IDF + LR/LGBM), CV evaluators
│   └── utils.py               # Metrics calculator & submission validator
├── submissions/               # Ready-to-submit CSV files
├── .env                       # Local secrets (ignored by Git)
├── .env.example               # Template for environment variables
├── AGENTS.md                  # This file
├── README.md                  # Team reference & schedule
└── requirements.txt           # Python dependencies
```

---

## 5. Submission Checklist
Before declaring any submission file ready:
1. [ ] Check file exists in `submissions/<name>_<timestamp>.csv`
2. [ ] Validate matching number of rows with `test.csv`
3. [ ] Check column headers: `id` and `target` (or competition exact headers)
4. [ ] Check `df.isna().sum().sum() == 0`
5. [ ] Confirm target range: `[0.0, 1.0]` for ROC-AUC / Log-Loss, or `{0, 1}` for F1 / Accuracy
