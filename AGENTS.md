# AGENTS.md — Critical Instructions for AI Coding Agents

Applies to Antigravity, Cursor, Copilot, Claude Code, Windsurf, and any
automated AI coding assistant.

## 1. Hackathon Context & Official Ground Truth

- **Competition:** `rmit-hackathon-2026` — **Challenge 1: Multilingual Prompt-Attack Classification** (Saigon South Campus, Kaggle).
- **Task Formulation:** **3-Class Classification** (`label` column):
  - `BENIGN` (~62% of train): Normal, safe user requests.
  - `JAILBREAK` (~31% of train): Adversarial manipulation attempting to bypass safety guardrails (DAN personas, roleplay, hypothetical scenarios, obfuscation).
  - `PROMPT_INJECTION` (~7% of train — **critical minority class**): Instruction hijacking attempting to override system prompts, inject control delimiters, or divert application control flow.
- **Target Languages:** **Portuguese (`pt`), French (`fr`), Italian (`it`), and English (`en`)** (Romance languages + English), explicitly indicated via the `language` column in both train and test sets. *(Note: Not Vietnamese or Malay).*
- **Evaluation Metric:** **Macro-averaged F1 Score** ($\text{Macro-F1} = \frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} + F1_{\text{PROMPT\_INJECTION}})$). Because `PROMPT_INJECTION` is only 7% of the dataset but accounts for **1/3 of the total score**, handling class imbalance (`class_weight="balanced"`, weighted loss, and per-class probability threshold/multiplier tuning) is paramount.
- **Dataset Schema:**
  - `train.csv` (14,000 rows): `id, text, language, label`
  - `test.csv` (6,000 rows): `id, text, language`
  - `sample_submission.csv` (6,000 rows): `id, label` (Exact string values: `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`).
  - **Text length nuance:** Prompts range from short snippets up to **~18,000 characters**.
- **Critical Competition Constraints:**
  - **Submission Quota:** **Maximum 6 attempts total** for the entire competition. Every submission must be verified on 5-Fold CV and approved by the Team Lead (Nhật Tân).
  - **Approved Models Only:** Any pretrained neural backbone **must** come from the official Kaggle Models whitelist approved by the organizers:
    - `protectai/deberta-v3-base-prompt-injection-v2`
    - `FacebookAI/xlm-roberta-base`
    - `patronus-studio/wolf-defender-prompt-injection`
    - `distilbert-base-multilingual-cased`
    - *(Classical ML models built from scratch on competition data such as `TfidfVectorizer` + `LogisticRegression`, `LinearSVC`, `LightGBM`, `CatBoost` are also standard baseline/ensemble components).*
  - **Submission Notebook Naming:** `<university>_<campus>_<team>_C1` (e.g., `RMIT_SGS_<team>_C1`).

## 2. Core Philosophy: "Baseline First, CV-Driven Ensembling"

1. **Protect the 6-submission quota:** Never submit unverified files. All modeling decisions must be validated locally or on Kaggle GPU using the shared 5-Fold split (`data/processed/folds.csv`) measuring **Out-Of-Fold (OOF) Macro-F1**.
2. **Start simple & fast:** Benchmark 3-class TF-IDF (`word` n-grams `(1, 3)` + `char_wb` n-grams `(2, 5)`) + `language` features + `LogisticRegression` / `LinearSVC` / `LightGBM` (`class_weight="balanced"`).
3. **Never freeze the local machine:** Do not launch heavy deep learning training jobs on local CPU/laptop GPUs without explicit user instruction. Run fine-tuning of approved Hugging Face transformers (`protectai/deberta-v3-base-prompt-injection-v2`, `FacebookAI/xlm-roberta-base`, etc.) on **Kaggle Notebook GPUs (T4 x2 / P100)**.

## 3. Strict Operating Rules (Zero Exceptions)

- **Never modify or overwrite `data/raw/`.** It is strictly read-only. Save cleaned, tokenized, fold-split, or engineered datasets under `data/processed/`.
- **Preserve Romance Diacritics and Adversarial Unicode.** Never use destructive ASCII-only sanitization such as `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. It strips French/Portuguese/Italian diacritics (`é, è, ê, ç, ã, õ, á, í, ó, ú, à, ù...`) and destroys adversarial punctuation used in prompt injection (e.g., Markdown/XML delimiters ` ``` `, `<|...|>` , `###`, Base64 padding `==`). Always use `src/data_loader.py:clean_text_safety()`.
- **Never hardcode API keys or secrets.** Retrieve keys via `os.getenv()` from `.env` locally or `kaggle_secrets` in a Kaggle Notebook. Never commit `.env` or `kaggle.json`.
- **Never edit another member's `.ipynb` file.** To prevent Git merge conflicts in notebook JSON metadata, each member maintains their own notebook (e.g., `notebooks/01_<name>_<task>.ipynb`). Refactor reusable logic into `src/*.py`.
- **Always validate `submission.csv` before saving.** Use `src/utils.py:validate_submission()`. Verify exact column names (`id`, `label`), exact row count matching `test.csv` (`6,000` rows), absence of `NaN`/`null`/`inf`, and that every prediction belongs strictly to `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`.

## 4. Team Members, Roles & Agent Guidance

When a user identifies themselves or works on a specific task, the AI Agent **must** tailor its technical advice, code suggestions, and scope according to their designated role.

### 1. Nhật Tân — Team Lead, Kaggle Pipeline & Ensemble Master

- **Scope and responsibilities**
  - **Quota & pipeline governance:** Strictly guards the **6 total submissions** limit for the entire competition; generates and locks `data/processed/folds.csv` (`src/folds.py:make_folds()`) so all 4 members evaluate on identical 5-Fold splits; validates every candidate submission with `src/utils.py:validate_submission()`.
  - **3-Class Fast Baseline:** Builds and verifies the 3-class baseline (`TF-IDF + language + LogisticRegression / Calibrated LinearSVC`, `class_weight="balanced"`) to establish the initial OOF Macro-F1 benchmark.
  - **Complementary model track:** Explores feature/embedding extraction from Approved Models combined with linear/MLP/CatBoost heads for ensemble diversity.
  - **Error analysis & hard example mining:** Inspects confusion matrices (specifically `JAILBREAK` vs `PROMPT_INJECTION` false positives/negatives across `pt`, `fr`, `it`, `en`), feeding actionable insights back to Cường Thịnh and Khánh Linh.
  - **3-Class Probability Blending & Macro-F1 Threshold Optimization:** Combines $(N \times 3)$ OOF and test probability matrices (`src/blend.py:blend()`) and optimizes per-class probability multipliers (`src/blend.py:optimize_class_multipliers()`) to maximize Macro-F1, especially boosting recall/precision on the 7% `PROMPT_INJECTION` class.
- **Agent instructions**
  - Provide end-to-end code for 3-class baselines, fold generation, OOF confusion matrix diagnostics, $(N \times 3)$ probability blending, and Dirichlet/Nelder-Mead class multiplier optimization for Macro-F1.
  - Assist with Kaggle CLI commands, notebook naming (`RMIT_SGS_<team>_C1`), and safe Git merges.

### 2. Trung Quân — Deep Learning & Approved Hugging Face Specialist (Kaggle GPU)

- **Scope and responsibilities**
  - Fine-tunes strictly **Approved Models** from the official Kaggle whitelist:
    - `protectai/deberta-v3-base-prompt-injection-v2`
    - `FacebookAI/xlm-roberta-base`
    - `patronus-studio/wolf-defender-prompt-injection`
    - `distilbert-base-multilingual-cased`
  - Works on **Kaggle Notebooks with GPU (T4 x2 / P100)** using the shared `folds.csv` split.
  - Mitigates the 62%–31%–7% class imbalance using **Class-Weighted CrossEntropyLoss** or **Multi-class Focal Loss**.
  - Handles long prompts (up to 18k chars) via **Head+Tail Token Truncation** (e.g., first 256 + last 256 tokens) and prepends the `language` tag (`"lang: fr | ..."`).
  - Exports 3-class OOF `(14000, 3)` and Test `(6000, 3)` probability arrays (`.npy`/`.parquet`) in canonical class order `['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']` for Nhật Tân to ensemble.
- **Agent instructions**
  - Provide clean, memory-efficient PyTorch / Hugging Face `transformers` code (`AutoModelForSequenceClassification` with `num_labels=3`, `LABEL2ID`, `ID2LABEL`, FP16 mixed precision, cosine scheduler, gradient accumulation).
  - Never suggest unapproved external backbones or run heavy neural training on the local CPU.

### 3. Cường Thịnh — Multilingual Security & NLP Feature Engineering Specialist

- **Scope and responsibilities**
  - Analyzes adversarial patterns across **Portuguese (`pt`), French (`fr`), Italian (`it`), and English (`en`)**.
  - Crafts regex and heuristic indicators that distinguish **`PROMPT_INJECTION`** (instruction override, system prompt extraction, delimiter escapes `<|im_start|>`, `### Instruction`, task hijacking) from **`JAILBREAK`** (DAN personas, hypothetical/roleplay framing, moral bypass, Base64/Hex/leetspeak obfuscation) in all 4 languages.
  - Exploits the `language` column (one-hot encoding, language-specific length/character statistics) and handles **long-text anomalies** (prompts up to 18,000 chars: log length, head vs. tail symbol density, newline bursts).
  - Delivers modular numeric/boolean feature DataFrames (`src/data_loader.py:extract_adversarial_features()`) to Khánh Linh.
- **Agent instructions**
  - Provide Unicode-safe regex patterns for `pt, fr, it, en` (refer to `docs/cheat_sheet_security_low_resource.md`), structural anomaly extractors for 18k-char texts, and vectorized pandas feature functions.

### 4. Khánh Linh — Tabular ML & Multiclass TF-IDF Specialist

- **Scope and responsibilities**
  - Builds fast, high-performing local 3-class models: `LightGBM` (`objective="multiclass"`, `num_class=3`, `class_weight="balanced"`), `CatBoost`, `LogisticRegression`, and `CalibratedClassifierCV(LinearSVC)`.
  - Combines multilingual TF-IDF matrices (word n-grams `(1, 3)` + char n-grams `(2, 5)` with `char_wb`), `language` indicators, and Cường Thịnh's tabular security features (via `scipy.sparse.hstack` / `ColumnTransformer`).
  - Evaluates strictly on the shared 5-Fold Stratified CV (`folds.csv`), tracking **Macro-F1** and per-class F1 (`BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`).
  - Exports 3-class OOF `(14000, 3)` and Test `(6000, 3)` probabilities plus validated `id, label` submissions to `submissions/`.
- **Agent instructions**
  - Provide clean scikit-learn / LightGBM multiclass pipelines, sparse + dense feature concatenation recipes, hyperparameter tuning for imbalanced 3-class tabular data, and feature importance diagnostics.

---

## 5. Codebase Architecture

```text
rmit-hackathon-2026/
├── docs/                      # Rules, booklet, multilingual security cheat sheet, prompts
├── data/
│   ├── raw/                   # Raw Kaggle data: train.csv, test.csv (STRICTLY READ-ONLY)
│   └── processed/             # Shared folds.csv and engineered datasets
├── notebooks/                 # Individual experiment notebooks
│                              # (e.g., 00_baseline_demo.ipynb, 01_<name>_<task>.ipynb)
├── src/                       # Shared team Python modules
│   ├── config.py              # Directory paths, seeds, environment helpers
│   ├── data_loader.py         # Safe loader, NFC cleaner, multilingual/long-text features
│   ├── folds.py               # Shared 5-Fold Stratified/Group split generator
│   ├── models.py              # 3-Class TF-IDF + language + LR/LinearSVC/LGBM pipelines
│   ├── blend.py               # (N x 3) probability blending & Macro-F1 multiplier tuning
│   └── utils.py               # 3-Class Macro-F1 metrics & strict submission validator
├── submissions/               # Validated ready-to-submit CSV files (id, label)
├── .env                       # Local secrets (ignored by Git)
├── .env.example               # Environment variable template
├── AGENTS.md                  # This file
├── prompt.md                  # Role-specific prompt templates for team members
├── README.md                  # Team reference, role table, and schedule
└── requirements.txt           # Python dependencies
```

## 6. Submission Checklist (Strictly Enforced — Max 6 Attempts Total!)

Before declaring any submission file ready or using one of the 6 Kaggle submission attempts:

1. [ ] Check that the file is saved in `submissions/<name>.csv` and notebook is named `<university>_<campus>_<team>_C1` when submitting via Kaggle Kernel.
2. [ ] Validate that its row count matches `test.csv` (**exact 6,000 rows**).
3. [ ] Check exact column headers: **`id` and `label`** (NOT `target`).
4. [ ] Check that `df.isna().sum().sum() == 0` and IDs match `test.csv["id"]` in exact order.
5. [ ] Confirm every value in `label` is a valid string in **`{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`** (never submit raw floats or integer codes `0/1/2` to Kaggle).
6. [ ] Inspect predicted class distribution (`sub_df["label"].value_counts(normalize=True)`) to ensure `PROMPT_INJECTION` is reasonably represented (typically ~5%–12%) and not collapsed to 0%.
