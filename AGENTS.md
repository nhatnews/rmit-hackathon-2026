# AGENTS.md — Critical Instructions for AI Coding Agents

Applies to Antigravity, Cursor, Copilot, Claude Code, Windsurf, and all team members.

## 1. Hackathon Context & Official Ground Truth

- **Competition:** `rmit-hackathon-2026` — **Challenge 1: Multilingual Prompt-Attack Classification** (Saigon South Campus, Kaggle).
- **Task formulation:** **3-class classification** (`label` column):
  - `BENIGN` (~62% of train): Normal, safe user requests.
  - `JAILBREAK` (~31% of train): Adversarial manipulation attempting to bypass safety guardrails (DAN personas, roleplay, hypothetical scenarios, obfuscation).
  - `PROMPT_INJECTION` (~7% of train — **critical minority class**): Instruction hijacking attempting to override system prompts, inject control delimiters, or divert application control flow.
- **Target languages:** **Portuguese (`pt` - 45%), French (`fr` - 25%), Italian (`it` - 20%), and English (`en` - 10%)** (Romance languages + English), explicitly indicated via the `language` column in both train and test sets.
- **Evaluation metric:** **Macro-averaged F1 score**
  $$\text{Macro-F1} = \frac{1}{3}(F1_{\text{BENIGN}} + F1_{\text{JAILBREAK}} + F1_{\text{PROMPT\_INJECTION}})$$
  Because `PROMPT_INJECTION` is only 7% of the dataset but accounts for **1/3 of the total score**, handling class imbalance (`class_weight="balanced"`, weighted loss, or per-class threshold multipliers) is paramount.
- **Dataset Schema:**
  - `train.csv` (14,000 rows): `id`, `text`, `language`, `label`
  - `test.csv` (6,000 rows): `id`, `text`, `language`
  - `sample_submission.csv` (6,000 rows): `id`, `label` (exact string values: `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`).
  - **Text length:** Prompts range from short snippets up to **~18,550 characters**.
- **Critical Competition Constraints:**
  - **Submission quota:** **Maximum 6 attempts total** for the entire competition across the whole team. Never submit unverified files.
  - **Approved models only:** Any pretrained neural backbone **must** come from the official Kaggle Models whitelist approved by the organizers. **No other pretrained backbone is allowed.**
    1. **Safety-Tuned Encoders (High Priority for Prompt Injection):**
       - `protectai/deberta-v3-base-prompt-injection-v2` — pre-trained specifically on prompt injection & jailbreak.
       - `patronus-studio/wolf-defender-prompt-injection` — security-tuned safety classifier.
    2. **Multilingual Encoders (High Priority for Romance Languages):**
       - `FacebookAI/xlm-roberta-base` — strong, balanced multilingual encoder (reliable, fast).
       - `FacebookAI/xlm-roberta-large` — strongest multilingual representation; requires batch size 4-8 + FP16.
       - `jhu-clsp/mmBERT-base` — fast, modern multilingual encoder.
       - `jhu-clsp/mmBERT-small` — ultra-fast multilingual encoder.
       - `distilbert/distilbert-base-multilingual-cased` — fastest transformer baseline.
       - `google/rembert` — 32-layer multilingual architecture; recommend batch size 8.
    3. **LLMs (LoRA / Fine-Tuning):**
       - `ibm-granite/granite-3.3-2b-instruct` — 2B parameter instruct model.
       - `mistralai/Mistral-7B-Instruct-v0.3` — 7B parameter instruct model.
    4. **Classical Tabular / Linear ML:**
       - `TF-IDF + Ridge/LinearSVC/LightGBM` (baseline provided in `src/models.py`, OOF Macro-F1 = `0.9332`).
  - **Notebook requirement:** State the chosen model at the top of the Kaggle notebook (e.g. `# Model: protectai/deberta-v3-base-prompt-injection-v2`).
  - **Notebook naming:** `<university>_<campus>_<team>_C1` (e.g. `RMIT_SGS_<team>_C1`).

---

## 2. Core Operating Model: Parallel Model Exploration with Strict Quota Governance

1. **Parallel Model Building:**
   - Any team member can choose and fine-tune any model from the official 10 Approved Models list.
   - All models use the unified starter template (`notebooks/template_train_approved_model.ipynb`) with plug-and-play `MODEL_NAME`.
2. **Unified Evaluation Benchmark (Fair Comparison):**
   - Every model **must** evaluate on the shared 5-Fold split (`data/processed/folds.csv`).
   - Checkpoints and model decisions are ranked strictly by **Macro-F1** (not accuracy or cross-entropy loss).
3. **Strict Quota Gatekeeper (Nhật Tân):**
   - All submissions to Kaggle are reviewed and submitted by **Nhật Tân**.
   - Before submission, verify:
     - `src/utils.py:validate_submission(sub_df)` passes with (6000, 2) shape and valid labels.
     - Label distribution on `test.csv`: `BENIGN` ~58-65%, `JAILBREAK` ~28-34%, `PROMPT_INJECTION` ~5-10%.
     - Target allocation: Max 3-4 submissions during Session 1 (morning), reserving at least 2 submissions for final optimizations in the afternoon.

---

## 3. Strict Technical Guardrails (Zero Exceptions)

- **Never modify or overwrite `data/raw/`.** It is strictly read-only.
- **Preserve Romance diacritics and adversarial Unicode.** Never use destructive ASCII regex such as `re.sub(r'[^a-zA-Z0-9]', ' ', text)`. Always use `src/data_loader.py:clean_text_safety()`.
- **Never hardcode secrets.** Retrieve tokens via `os.getenv()` or `kaggle_secrets`.
- **Input Text Formatting:** Use prefix `f"lang: {language} | {clean_text}"` to activate language-specific attention heads.
- **Head + Tail Token Truncation:** For long texts (up to 18.5k chars), take the first 256 tokens + last 256 tokens (= 512 total) to capture both opening framing and trailing attack payloads.
- **Handle Class Imbalance:** Apply Class-Weighted CrossEntropyLoss (`weights = torch.tensor([1.0, 2.0, 8.8])`) or post-processing probability multipliers (`optimize_class_multipliers()`) to prevent the 7% `PROMPT_INJECTION` class from collapsing.

---

## 4. Submission Checklist (Before Any Kaggle Attempt)

1. [ ] Saved in `submissions/<name>.csv` with exact columns: `id` and `label`.
2. [ ] Row count matches `test.csv` (**exactly 6,000 rows**).
3. [ ] No `NaN`, null, or empty string values.
4. [ ] Every label belongs to `{'BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'}` (no integer codes `0, 1, 2`).
5. [ ] Predicted distribution contains reasonable `PROMPT_INJECTION` (~5%–10%).
6. [ ] Approved by Nhật Tân and recorded in the submission log.
