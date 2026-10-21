# AGENTS.md — Critical Instructions for AI Coding Agents

Applies to Antigravity, Cursor, Copilot, Claude Code, Windsurf, and any
automated AI coding assistant.

## 1. Hackathon Context & Official Ground Truth

- **Competition:** `rmit-hackathon-2026` — **Challenge 1: Multilingual
  Prompt-Attack Classification** (Saigon South Campus, Kaggle).
- **Task formulation:** **3-class classification** (`label` column):
  - `BENIGN` (~62% of train): Normal, safe user requests.
  - `JAILBREAK` (~31% of train): Adversarial manipulation attempting to bypass safety guardrails (DAN personas, roleplay, hypothetical scenarios, obfuscation).
  - `PROMPT_INJECTION` (~7% of train — **critical minority class**): Instruction hijacking attempting to override system prompts, inject control delimiters, or divert application control flow.
- **Target languages:** **Portuguese (`pt`), French (`fr`), Italian (`it`),
  and English (`en`)** (Romance languages + English), explicitly indicated via
  the `language` column in both train and test sets. *(Note: Not Vietnamese or
  Malay.)*
- **Evaluation metric:** **Macro-averaged F1 score**
  ($\text{Macro-F1} = \frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} +
  F1_{\text{PROMPT\_INJECTION}})$). Because `PROMPT_INJECTION` is only 7% of
  the dataset but accounts for **1/3 of the total score**, handling class
  imbalance (`class_weight="balanced"`, weighted loss, and per-class
  probability threshold/multiplier tuning) is paramount.
- **Dataset Schema:**
  - `train.csv` (14,000 rows): `id`, `text`, `language`, `label`
  - `test.csv` (6,000 rows): `id`, `text`, `language`
  - `sample_submission.csv` (6,000 rows): `id`, `label` (exact string
    values: `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`).
  - **Text length:** Prompts range from short snippets up to **~18,000
    characters**.
- **Critical Competition Constraints:**
  - **Submission quota:** **Maximum 6 attempts total** for the entire
    competition. Every submission must be verified on 5-Fold CV and approved
    by the Team Lead (Nhật Tân).
  - **Approved models only:** Any pretrained neural backbone **must** come
    from the official Kaggle Models whitelist approved by the organizers.
    **No other pretrained backbone is allowed.**
    - **Encoders (fast and multilingual):**
      - `distilbert/distilbert-base-multilingual-cased` — fastest baseline.
      - `jhu-clsp/mmBERT-small` — fast, multilingual.
      - `jhu-clsp/mmBERT-base` — strong multilingual.
      - `FacebookAI/xlm-roberta-base` — strong multilingual.
      - `FacebookAI/xlm-roberta-large` — best encoder; high VRAM.
      - `google/rembert` — multilingual; use batch size 8.
    - **Encoders (safety-tuned for prompt injection):**
      - `protectai/deberta-v3-base-prompt-injection-v2` — already trained for
        prompt injection.
      - `patronus-studio/wolf-defender-prompt-injection` — already trained for
        prompt injection.
    - **LLMs (LoRA / fine-tuning):**
      - `ibm-granite/granite-3.3-2b-instruct` — LLM fine-tuning / LoRA.
      - `mistralai/Mistral-7B-Instruct-v0.3` — LLM fine-tuning / LoRA.
    - Classical ML models built from competition data, such as
      `TfidfVectorizer` + `LogisticRegression`, `LinearSVC`, `LightGBM`, or
      `CatBoost`, are also standard baseline/ensemble components.
  - **Notebook requirement:** State the chosen model at the top of the Kaggle
    notebook.
  - **Notebook naming:** `<university>_<campus>_<team>_C1` (e.g.,
    `RMIT_SGS_<team>_C1`).

## 2. Core Philosophy: "Baseline First, CV-Driven Ensembling"

1. **Protect the 6-submission quota:** Never submit unverified files. Validate
   all modeling decisions locally or on a Kaggle GPU using the shared 5-Fold
   split (`data/processed/folds.csv`) and **Out-Of-Fold (OOF) Macro-F1**.
2. **Start simple and fast:** Benchmark 3-class TF-IDF (`word` n-grams
   `(1, 3)` + `char_wb` n-grams `(2, 5)`) with `language` features and
   `LogisticRegression`, `LinearSVC`, or `LightGBM`
   (`class_weight="balanced"`).
3. **Keep the local machine responsive:** Do not launch heavy deep learning
   training jobs on local CPUs or laptop GPUs without explicit user
   instruction. Fine-tune approved Hugging Face transformers (e.g.,
   `protectai/deberta-v3-base-prompt-injection-v2` and
   `FacebookAI/xlm-roberta-base`) on **Kaggle Notebook GPUs (T4 x2 / P100)**.

## 3. Strict Operating Rules (Zero Exceptions)

- **Never modify or overwrite `data/raw/`.** It is strictly read-only. Save cleaned, tokenized, fold-split, or engineered datasets under `data/processed/`.
- **Preserve Romance diacritics and adversarial Unicode.** Never use
  destructive ASCII-only sanitization such as
  `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. It strips French, Portuguese, and
  Italian diacritics (`é, è, ê, ç, ã, õ, á, í, ó, ú, à, ù...`) and destroys
  adversarial punctuation used in prompt injection (e.g., Markdown/XML
  delimiters ` ``` `, `<|...|>`, `###`, and Base64 padding `==`). Always use
  `src/data_loader.py:clean_text_safety()`.
- **Never hardcode API keys or secrets.** Retrieve keys via `os.getenv()` from `.env` locally or `kaggle_secrets` in a Kaggle Notebook. Never commit `.env` or `kaggle.json`.
- **Never edit another member's `.ipynb` file.** To prevent Git merge conflicts in notebook JSON metadata, each member maintains their own notebook (e.g., `notebooks/01_<name>_<task>.ipynb`). Refactor reusable logic into `src/*.py`.
- **Always validate submissions before saving.** Use
  `src/utils.py:validate_submission()`. Verify the exact column names (`id`,
  `label`), row count matching `test.csv` (6,000 rows), absence of
  `NaN`/`null`/`inf`, and that every prediction belongs to
  `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`.

## 4. Team Members, Roles & Agent Guidance

When a user identifies themselves or works on a specific task, the AI Agent
**must** tailor its technical advice, code suggestions, and scope according to
their designated role.

### 1. Nhật Tân — Team Lead, Kaggle Pipeline & Ensemble Master

- **Scope and responsibilities**
  - **Quota and pipeline governance:** Strictly guards the **6 total
    submissions** limit for the entire competition; generates and locks
    `data/processed/folds.csv` (`src/folds.py:make_folds()`) so all four
    members evaluate on identical 5-Fold splits; validates every candidate
    submission with `src/utils.py:validate_submission()`.
  - **3-Class Fast Baseline:** Builds and verifies the 3-class baseline (`TF-IDF + language + LogisticRegression / Calibrated LinearSVC`, `class_weight="balanced"`, OOF Macro-F1 = `0.9332`).
  - **Fast Approved Multilingual Encoders & Embeddings:** Runs fast encoders from the 10 Approved Models whitelist (e.g., `jhu-clsp/mmBERT-base`, `jhu-clsp/mmBERT-small`, `distilbert/distilbert-base-multilingual-cased`) on a parallel Kaggle GPU session or extracts embeddings for linear/MLP/CatBoost heads to prevent GPU bottlenecks on a single person.
  - **Error analysis and hard example mining:** Inspects OOF confusion matrices across `pt`, `fr`, `it`, and `en` (focusing on the #1 error bottleneck: `BENIGN` vs. `JAILBREAK` in Portuguese `pt` due to contrastive benign roleplay vs. subtle academic jailbreaks, and #2: `BENIGN` vs. `PROMPT_INJECTION` in `en, it, fr`), then shares actionable insights with Cường Thịnh and Khánh Linh.
  - **3-class probability blending and Per-Language Macro-F1 optimization:**
    Combines $(N \times 3)$ OOF and test probability matrices
    (`src/blend.py:blend()`) and optimizes both global (`optimize_class_multipliers()`) and per-language probability multipliers (`optimize_per_language_multipliers()`) to maximize Macro-F1.
- **Agent instructions**
  - Provide end-to-end code for 3-class baselines, fold generation, OOF
    confusion matrix diagnostics, $(N \times 3)$ probability blending, and
    global/per-language class multiplier optimization for Macro-F1.
  - Assist with Kaggle CLI commands, notebook naming
    (`RMIT_SGS_<team>_C1`), and safe Git merges.

### 2. Trung Quân — Deep Learning & Approved Hugging Face Specialist (Kaggle GPU)

- **Scope and responsibilities**
  - Fine-tunes strictly **Approved Models** from the official 10-model Kaggle
    whitelist (prioritizing safety-tuned & heavy multilingual models:
    `protectai/deberta-v3-base-prompt-injection-v2`,
    `patronus-studio/wolf-defender-prompt-injection`,
    `FacebookAI/xlm-roberta-large`, `FacebookAI/xlm-roberta-base`,
    `google/rembert` with batch size 8, or LoRA on `ibm-granite/granite-3.3-2b-instruct` / `mistralai/Mistral-7B-Instruct-v0.3`).
  - Always states the chosen model at the top of the Kaggle notebook per competition rules.
  - Works on **Kaggle Notebooks with GPU (T4 x2 / P100)** using the shared `folds.csv` split.
  - Mitigates the 62%–31%–7% class imbalance using **Class-Weighted CrossEntropyLoss** or **Multi-class Focal Loss**, while leveraging Transformer self-attention to resolve Portuguese (`pt`) contrastive `BENIGN` vs. subtle `JAILBREAK` errors.
  - Handles long prompts (up to 18k chars) via **Head+Tail Token Truncation** (e.g., first 256 + last 256 tokens) and prepends the `language` tag (`"lang: fr | ..."`).
  - Exports 3-class OOF `(14000, 3)` and Test `(6000, 3)` probability arrays
    (`.npy`/`.parquet`) in canonical class order
    `['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']` for Nhật Tân to ensemble.
- **Agent instructions**
  - Provide clean, memory-efficient PyTorch / Hugging Face `transformers`
    code (`AutoModelForSequenceClassification` with `num_labels=3`, `LABEL2ID`,
    `ID2LABEL`, FP16 mixed precision, cosine scheduler, and gradient
    accumulation).
  - Never suggest unapproved external backbones or run heavy neural training on the local CPU.

### 3. Cường Thịnh — Multilingual Security & NLP Feature Engineering Specialist

- **Scope and responsibilities**
  - Analyzes real OOF error patterns across **Portuguese (`pt` — 45%), French (`fr` — 25%), Italian (`it` — 20%), and English (`en` — 10%)**.
  - Focuses on the two primary error sources found in EDA:
    1. **`BENIGN` vs. `JAILBREAK` in Portuguese (`pt` — 80% of all errors):** Crafts semantic payload vs. benign roleplay indicators to distinguish *Contrastive Benign* (harmless requests wrapped in playful/adversarial framing) from *Subtle Jailbreaks* (harmful requests wrapped in polite academic/historical/hypothetical framing).
    2. **`BENIGN` vs. `PROMPT_INJECTION` in `en, it, fr` (18% of all errors):** Crafts short-prompt command/override and delimiter escape detectors (`<|im_start|>`, `### Instruction`, system prompt extraction).
  - Exploits **Golden Length Boundaries** discovered in `train.csv` (`char_len < 112` $\rightarrow$ 0% `JAILBREAK`; `char_len > 3752` $\rightarrow$ 0% `JAILBREAK`; `char_len > 5128` $\rightarrow$ 100% `BENIGN`) and per-language statistics.
  - Delivers modular numeric/boolean feature DataFrames (`src/data_loader.py:extract_adversarial_features()`) to Khánh Linh.
- **Agent instructions**
  - Provide Unicode-safe regex patterns for `pt, fr, it, en` (refer to `docs/cheat_sheet_security_low_resource.md`), golden length boundary extractors, and vectorized pandas feature functions.

### 4. Khánh Linh — Tabular ML & Multiclass TF-IDF Specialist

- **Scope and responsibilities**
  - Builds fast, high-performing local 3-class models: `LightGBM`
    (`objective="multiclass"`, `num_class=3`,
    `class_weight="balanced"`), `CatBoost`, `LogisticRegression`, and
    `CalibratedClassifierCV(LinearSVC)`.
  - Combines multilingual TF-IDF matrices (word n-grams `(1, 3)` and char
    n-grams `(2, 5)` with `char_wb`), `language` indicators, OOF meta-probabilities (stacking), and Cường Thịnh's
    tabular security features (via `scipy.sparse.hstack` / `TruncatedSVD`).
  - Explores **Per-Language Specialist Models** (especially a dedicated Portuguese `pt` classifier for the 45% `pt` subset where `BENIGN` vs. `JAILBREAK` is hardest).
  - Evaluates strictly on the shared 5-Fold Stratified CV (`folds.csv`),
    tracking **Macro-F1** and per-class F1 (`BENIGN`, `JAILBREAK`,
    `PROMPT_INJECTION`).
  - Exports 3-class OOF `(14000, 3)` and Test `(6000, 3)` probabilities,
    plus validated `id, label` submissions to `submissions/`.
- **Agent instructions**
  - Provide clean scikit-learn/LightGBM multiclass pipelines, per-language specialist training loops, sparse/dense
    feature concatenation recipes, and feature-importance diagnostics.

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

1. [ ] Check that the file is saved in `submissions/<name>.csv` and the
   notebook is named `<university>_<campus>_<team>_C1` when submitting via a
   Kaggle Kernel.
2. [ ] Validate that its row count matches `test.csv` (**exactly 6,000 rows**).
3. [ ] Check the exact column headers: **`id` and `label`** (not `target`).
4. [ ] Check that `df.isna().sum().sum() == 0` and IDs match
   `test.csv["id"]` in exact order.
5. [ ] Confirm every `label` value is a valid string in
   **`{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}`**. Never submit raw floats
   or integer codes `0/1/2` to Kaggle.
6. [ ] Inspect the predicted class distribution
   (`sub_df["label"].value_counts(normalize=True)`) to ensure
   `PROMPT_INJECTION` is reasonably represented (typically ~5%–12%) and not
   collapsed to 0%.
