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

## 2. Core Philosophy: "Single Approved Backbone on Kaggle (DeBERTa-v3 Focus)"

1. **Strict 100% Rule Compliance & Reproducibility:**
   - BTC Rule: *"You must build your solution on one of the following approved models... no other pretrained backbone is allowed"* and *"State the chosen model at the top of the Kaggle notebook"*.
   - The team focuses 100% of its resources on **one primary approved backbone**: **`protectai/deberta-v3-base-prompt-injection-v2`** (a security-tuned model already pretrained for prompt injection).
   - The complete pipeline runs end-to-end within the official Kaggle Notebook (`RMIT_SGS_<team>_C1`), generating `submission.csv` directly upon running the notebook, ensuring 100% audit compliance and reproducibility.
2. **Protect the 6-submission quota:** Never submit unverified files. Validate
   all modeling decisions using the shared 5-Fold split (`data/processed/folds.csv`) and **Out-Of-Fold (OOF) Macro-F1**.
3. **Local Machine vs. Kaggle GPU:**
   - Keep the local machine responsive: Baseline models (TF-IDF + Linear Classifier, OOF Macro-F1 = `0.9316`) run locally as a sanity check and safety net.
   - Deep Learning training runs strictly on **Kaggle Notebook GPUs (T4 x2 / P100)**.

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

## 4. Team Members, Roles & Agent Guidance (Single-Model Pipeline Collaboration)

When a user identifies themselves or works on a specific task, the AI Agent
**must** tailor its technical advice, code suggestions, and scope according to
their designated role within the unified Kaggle single-model pipeline.

### 1. Nhật Tân — Team Lead, Kaggle Pipeline & Quota Master

- **Scope and responsibilities**
  - **Quota and pipeline governance:** Strictly guards the **6 total
    submissions** limit for the entire competition; verifies the official Kaggle
    Notebook `RMIT_SGS_<team>_C1`; runs `src/utils.py:validate_submission()` on every output before submitting.
  - **In-Notebook Macro-F1 Threshold & Multiplier Optimization:**
    Embeds the post-processing logic (`optimize_class_multipliers()`) directly into the final cells of the Kaggle notebook to scale class probabilities $(m_0, m_1, m_2)$, boosting sensitivity for the 7% `PROMPT_INJECTION` minority class.
  - **Error Analysis and Iteration Direction:**
    Inspects OOF confusion matrices across `pt`, `fr`, `it`, and `en` (focusing on the #1 error bottleneck: `BENIGN` vs. `JAILBREAK` in Portuguese `pt` due to contrastive benign roleplay vs. subtle academic jailbreaks, and #2: `BENIGN` vs. `PROMPT_INJECTION` in `en, it, fr`), directing prompt/feature updates to Cường Thịnh and loss/hyperparameter updates to Khánh Linh.
- **Agent instructions**
  - Provide end-to-end code for notebook orchestration, threshold/multiplier tuning on PyTorch logits/softmax, Kaggle submission verification, and OOF error diagnostics.
  - Assist with Kaggle CLI commands, notebook naming (`RMIT_SGS_<team>_C1`), and safe Git merges.

### 2. Trung Quân — Kaggle GPU & Model Training Lead

- **Scope and responsibilities**
  - Builds and maintains the official Kaggle Notebook (`RMIT_SGS_<team>_C1`) powered by **`protectai/deberta-v3-base-prompt-injection-v2`** with `AutoModelForSequenceClassification` (`num_labels=3`).
  - Always declares `# Model: protectai/deberta-v3-base-prompt-injection-v2` at the very top of the notebook per competition rules.
  - Implements **Head + Tail Token Truncation** (e.g., first 256 + last 256 tokens = 512 max length) to capture both opening framing and trailing attack payloads in prompts up to 18k chars.
  - Enables FP16 mixed precision, Cosine learning rate scheduling with warmup, and gradient accumulation on Kaggle GPU (T4 x2).
  - Ensures the notebook executes end-to-end to output `/kaggle/working/submission.csv` seamlessly upon "Run All".
- **Agent instructions**
  - Provide clean, memory-efficient PyTorch / Hugging Face `transformers` code (`AutoModelForSequenceClassification`, `AutoTokenizer`, Head+Tail truncation, FP16, DataLoader, training loop).
  - Never run heavy neural training on the local CPU or suggest unapproved external backbones.

### 3. Cường Thịnh — Prompt Formatting & Text Engineering Specialist

- **Scope and responsibilities**
  - Prepares the optimal textual representation for DeBERTa-v3 across **Portuguese (`pt` — 45%), French (`fr` — 25%), Italian (`it` — 20%), and English (`en` — 10%)**.
  - Designs the input formatting prefix: `f"lang: {language} | {clean_text}"` to activate language-specific attention heads.
  - **Resolves the Portuguese (`pt`) bottleneck (80% of all errors):**
    Crafts text pre-formatting to help the transformer separate *Contrastive Benign* (playful/hypothetical framing with safe intent) from *Subtle Jailbreaks* (academic/historical persona masking harmful intent).
  - Integrates **Golden Length Boundary** cues into prompt text or metadata (`char_len < 112` or `> 3752` $\rightarrow$ impossible `JAILBREAK`; `char_len > 5128` $\rightarrow$ 100% `BENIGN`).
  - Ensures all text cleaning preserves Romance diacritics and attack delimiters via `clean_text_safety()`.
- **Agent instructions**
  - Provide vectorized pandas text formatting functions, Unicode-safe regex patterns (refer to `docs/cheat_sheet_security_low_resource.md`), and prompt engineering recipes for DeBERTa input pipelines.

### 4. Khánh Linh — Hyperparameters, Loss Tuning & Validation Specialist

- **Scope and responsibilities**
  - Mitigates the 62%–31%–7% class imbalance in DeBERTa-v3 training by implementing **Class-Weighted CrossEntropyLoss** (`weights = torch.tensor([1.0, 2.0, 8.8])`) or **Multi-class Focal Loss**.
  - Tunes training hyperparameters: Learning rate (`1.5e-5` – `2e-5`), warmup ratio (`0.1`), weight decay (`0.01`), batch size with gradient accumulation (`effective batch size 16-32`).
  - Evaluates models strictly on the shared 5-Fold split (`data/processed/folds.csv`), selecting the best epoch checkpoint by **Macro-F1** (not loss) to prevent leaderboard overfitting.
  - Maintains a standby configuration for the secondary multilingual backbone (`FacebookAI/xlm-roberta-base`) for comparative validation if needed.
- **Agent instructions**
  - Provide PyTorch loss functions (weighted CrossEntropy, Focal Loss), learning rate scheduler setups, stratified fold validation loops, and metric logging functions.

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
