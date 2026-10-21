# AGENTS.md — Critical Instructions for AI Coding Agents

Applies to Antigravity, Cursor, Copilot, Claude Code, Windsurf, and any
automated AI coding assistant.

## 1. Hackathon Context & Domain

- **Competition:** RMIT Hackathon 2026 (Saigon South Campus) on Kaggle.
- **Theme:** `Security x Generative AI x Low-Resource Languages`
  - AI safety and security: jailbreak detection, prompt injection,
    adversarial obfuscation, and red teaming.
  - Low-resource languages: Vietnamese, Malay, regional dialects, and
    code-switching (multilingual mix).
- **Time constraint:** 10:00 AM–6:00 PM (8-hour sprint). Every minute counts!

## 2. Core Philosophy: "Baseline First" (Speed & Stability)

1. **Ship a valid submission early:** Target submitting a valid file to Kaggle
   before 12:00 PM.
2. **Start simple:** Implement and benchmark standard baselines first:
   - Feature extraction: `TfidfVectorizer` (word-level and char n-grams
     `(2, 5)` to capture adversarial token splitting and typo obfuscation).
   - Classifiers: `LogisticRegression`, `LinearSVC`, or `LightGBM`.
   - Lightweight API-based evaluation when appropriate.
3. **Never freeze the local machine:** Do not launch long-running deep learning
   training jobs locally without explicit user instruction. Suggest a Kaggle
   Notebook GPU (P100 / T4) for heavy neural models (mDeBERTa, XLM-RoBERTa,
   etc.).

## 3. Strict Operating Rules (Zero Exceptions)

- **Never modify or overwrite `data/raw/`.** It is strictly read-only. Save
  cleaned, tokenized, augmented, or engineered data under `data/processed/`.
- **Preserve Vietnamese and Indigenous Unicode.** Never use generic
  sanitization such as `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. It strips
  Vietnamese diacritics (`á, ế, ơ, ư, đ...`), Malay text, and adversarial
  punctuation used in prompt injection (e.g., base64 delimiters, Markdown
  escapes, and zero-width characters). Use
  `src/data_loader.py:clean_text_safety()`.
- **Never hardcode API keys or secrets.** Retrieve keys via `os.getenv()` from
  `.env` locally or Kaggle Secrets in a Kaggle Notebook. Never commit `.env`
  or `kaggle.json`.
- **Never edit another member's `.ipynb` file.** To prevent Git merge
  conflicts in notebook JSON metadata, each member must maintain their own
  notebook (e.g., `notebooks/01_<name>_<task>.ipynb`). Refactor reusable
  logic into `src/*.py`.
- **Always validate `submission.csv` before saving.** Use
  `src/utils.py:validate_submission()`. Verify the column names (`id`,
  `target`), exact row count matching `test.csv`, absence of `NaN`, `null`,
  and `inf`, and the correct output type: probabilities in `[0, 1]` for
  ROC-AUC or binary labels `0/1` for F1/Accuracy.

## 4. Team Members, Roles & Agent Guidance

When a user identifies themselves or works on a specific task, the AI Agent
**must** tailor its technical advice, code suggestions, and scope according to
their designated role.

### 1. Nhật Tân — Team Lead, Kaggle Pipeline & Active Modeler

- **Scope and responsibilities**
  - **Team and pipeline management:** Manages Git merges, repository
    infrastructure, schedule adherence, and is the sole authorized submitter
    to Kaggle (enforces the 5 submissions/day limit and verifies files with
    `src/utils.py:validate_submission()`).
  - **Fast baseline (10:00–11:30):** Quickly prototypes and ships the first
    valid submission (TF-IDF + LogisticRegression / LinearSVC) to secure an
    early leaderboard score and relieve pressure on the team.
  - **Third model track (sentence embeddings / linear classifiers):** Explores
    complementary modeling tracks (e.g., multilingual sentence embeddings via
    `sentence-transformers`, fast LinearSVC, or CatBoost) to ensure ensemble
    diversity.
  - **Error analysis and hard example mining:** Inspects false positives and
    negatives from model outputs, discovers evasive teencode/adversarial
    patterns, and feeds actionable insights back to Cường Thịnh and Khánh
    Linh.
  - **Advanced ensembling and calibration:** Applies probability calibration
    (Platt scaling / Isotonic regression), rank averaging, and Optuna/Scipy
    weight optimization on Out-Of-Fold (OOF) predictions.
- **Agent instructions**
  - Provide end-to-end guidance from high-level pipeline strategy and
    submission validation to concrete code for fast baselines, sentence
    embedding pipelines, error analysis routines, and advanced
    blending/calibration.
  - Guide him on Kaggle CLI commands, quota management, and safe Git
    workflows.
  - Help him diagnose hard examples and guide his teammates with data-backed
    feedback.

### 2. Trung Quân — Deep Learning & Hugging Face Specialist

- **Scope and responsibilities**
  - Develops and fine-tunes pretrained multilingual transformers
    (`microsoft/mdeberta-v3-base`, `xlm-roberta-base`, `vinai/phobert-base-v2`).
  - Works primarily on **Kaggle Notebooks with GPU (T4 x2 / P100)** to avoid
    freezing local hardware.
  - Produces probability predictions from neural models ready for downstream
    ensembling.
- **Agent instructions**
  - Provide lightweight, idiomatic PyTorch / Hugging Face `transformers` code
    (`AutoModelForSequenceClassification`, `AutoTokenizer`, `Trainer` or
    PyTorch train loops).
  - Guide on handling token truncation (`max_length=256/512`), learning rate
    scheduling, mixed precision (FP16), and GPU memory optimization. Never run
    heavy local training.

### 3. Cường Thịnh — Security & NLP Feature Engineering Specialist

- **Scope and responsibilities**
  - Analyzes the dataset for adversarial jailbreak attacks (DAN personas,
    direct injections, obfuscations, character splitting, Base64/Hex ciphers,
    Vietnamese teencode).
  - Crafts domain-specific heuristic and statistical features (regex patterns,
    special char ratios, perplexity hints, Unicode NFC normalization).
  - Hands over engineered feature DataFrames to Khánh Linh for tabular
    modeling.
- **Agent instructions**
  - Provide regex utilities, string sanitizers that preserve Unicode, and
    feature extraction functions. Refer to
    `docs/cheat_sheet_security_low_resource.md`.
  - Help write modular feature functions that output numeric/boolean columns
    for easy concatenation into tabular models.

### 4. Khánh Linh — Machine Learning & TF-IDF / Tabular Specialist

- **Scope and responsibilities**
  - Builds fast, reliable local models: `LightGBM`, `CatBoost`,
    `LogisticRegression`, and `LinearSVC`.
  - Maximizes TF-IDF representations (word n-grams `(1, 2)` + char n-grams
    `(2, 5)` with `char_wb`).
  - Combines TF-IDF sparse matrices with tabular features from Cường Thịnh.
  - Implements rigorous Stratified 5-Fold Cross-Validation on local CPU.
- **Agent instructions**
  - Provide clean scikit-learn and LightGBM pipelines, K-Fold validation
    loops, hyperparameter tuning recipes, and feature importance diagnostics.
  - Ensure predictions are properly formatted (probabilities in `[0.0, 1.0]`)
    and exported to `submissions/` for Nhật Tân to validate.

---

## 5. Codebase Architecture

```text
rmit-hackathon-2026/
├── docs/                      # Rules, competition booklet, and prompt templates
├── data/
│   ├── raw/                   # Raw Kaggle data (STRICTLY READ-ONLY)
│   └── processed/             # Cleaned and engineered datasets
├── notebooks/                 # Individual experiment notebooks
│                              # (e.g., 01_<name>_<task>.ipynb)
├── src/                       # Shared team Python modules
│   ├── config.py              # Directory paths, seeds, environment helpers
│   ├── data_loader.py         # Safe loader and Unicode-preserving cleaner
│   ├── models.py              # TF-IDF + LR/LGBM baselines and CV evaluators
│   └── utils.py               # Metrics calculator and submission validator
├── submissions/               # Ready-to-submit CSV files
├── .env                       # Local secrets (ignored by Git)
├── .env.example               # Environment variable template
├── AGENTS.md                  # This file
├── README.md                  # Team reference and schedule
└── requirements.txt           # Python dependencies
```

## 6. Submission Checklist

Before declaring any submission file ready:

1. [ ] Check that the file exists in `submissions/<name>_<timestamp>.csv`.
2. [ ] Validate that its row count matches `test.csv`.
3. [ ] Check the column headers: `id` and `target` (or the competition's
   exact headers).
4. [ ] Check that `df.isna().sum().sum() == 0`.
5. [ ] Confirm the target range: `[0.0, 1.0]` for ROC-AUC/Log-Loss, or
   `{0, 1}` for F1/Accuracy.
