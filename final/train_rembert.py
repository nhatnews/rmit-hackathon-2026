"""
train_rembert.py
Script huấn luyện 5-Fold chuyên biệt cho google/rembert trên Kaggle GPU T4.
Tuân thủ nghiêm ngặt luật Single-Model và 100% Self-Contained Input:
- SampleWeightedLabelSmoothingLoss (khử độ lệch weight * label_smoothing).
- Intra-Fold Top-2 Snapshot Averaging (10 checkpoints trên 5 folds).
- Hỗ trợ cả 2 chế độ Head: 'standard' (V3) và 'hybrid_msd' (V4 - [CLS + Mean + Max] + 5x Multi-Sample Dropout).
- Tự động sinh 5-Fold StratifiedGroupKFold khớp 100% với folds.csv từ train.csv nếu không gắn file ngoài.
"""

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Chế độ chạy Offline hoàn toàn theo luật Kaggle
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = False

# Hằng số chuẩn bài toán
VALID_LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và torch trên cả CPU lẫn CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_msg(msg: str, log_file: Optional[Path] = None) -> None:
    """In log ra stdout có flush realtime và lưu tệp log nếu được chỉ định."""
    print(msg, flush=True)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def find_model_dir() -> Path:
    """
    Tìm thư mục chứa model RemBERT trên Kaggle Models (/kaggle/input/...).
    Đảm bảo chạy offline không phụ thuộc internet và đúng luật BTC.
    """
    search_dirs = [Path("/kaggle/input"), Path("models"), Path(".")]
    found_configs = []
    for base in search_dirs:
        if base.exists():
            for cfg in base.rglob("config.json"):
                found_configs.append(str(cfg))
                if "rembert" in str(cfg).lower():
                    return cfg.parent
                try:
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    if data.get("model_type") == "rembert":
                        return cfg.parent
                except Exception:
                    pass
    raise FileNotFoundError(
        "Không tìm thấy model RemBERT! Vui lòng Add Input: "
        "Models -> aisuko/rembert -> Transformers -> default. "
        f"Các config.json tìm thấy: {found_configs}"
    )


def find_file(name: str, required: bool = True) -> Optional[Path]:
    """Tìm file dữ liệu trong /kaggle/input hoặc các thư mục dữ liệu cục bộ."""
    search_dirs = [Path("/kaggle/input"), Path("data"), Path(".")]
    for base in search_dirs:
        if base.exists():
            for f in base.rglob(name):
                if f.is_file():
                    return f
    if required:
        raise FileNotFoundError(f"Không tìm thấy tệp bắt buộc: '{name}'. Vui lòng kiểm tra lại Input!")
    return None


def clean_text_safety(text: Any) -> str:
    """
    Chuẩn hóa văn bản an toàn cho tấn công đa ngôn ngữ (pt, fr, it, en).
    BẢO TỒN CHỮ HOA/THƯỜNG (preserve_case=True) và DẤU ROMANCE.
    """
    if not isinstance(text, str):
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def load_or_generate_folds(train_df: pd.DataFrame, log_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Tự động đọc folds.csv nếu có, hoặc sinh trực tiếp từ train.csv bằng StratifiedGroupKFold(seed=42)
    đảm bảo khớp 100% từng dòng (14,000/14,000) mà không cần phụ thuộc bất kỳ dataset phụ nào bên ngoài.
    """
    key = train_df["text"].fillna("").astype(str).map(
        lambda s: re.sub(r"\s+", " ", clean_text_safety(s).lower())  # V5: .lower() khớp folds.csv
    )
    groups = pd.factorize(key)[0]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    gen_folds = np.full(len(train_df), -1, dtype=int)
    for f_idx, (_, va_idx) in enumerate(sgkf.split(train_df, train_df["label"].values, groups)):
        gen_folds[va_idx] = f_idx

    folds_path = find_file("folds.csv", required=False)
    if folds_path is not None:
        folds_df = pd.read_csv(folds_path, dtype={"id": str})
        merged = train_df.merge(folds_df, on="id", how="left", validate="1:1")
        assert merged["fold"].notna().all(), "LỖI: folds.csv không khớp 1:1 với train.csv!"
        merged["fold"] = merged["fold"].astype(int)
        matches = bool((merged["fold"].values == gen_folds).all())
        log_msg(f"[*] Đã đọc folds.csv từ {folds_path} (Khớp StratifiedGroupKFold nội bộ: {matches})", log_file)
        return merged
    else:
        log_msg("[*] Không gắn folds.csv ngoài -> Tự sinh 5-Fold StratifiedGroupKFold(seed=42) trực tiếp từ train.csv!", log_file)
        out_df = train_df.copy()
        out_df["fold"] = gen_folds
        return out_df


def prepare_input_ids(texts: List[str], tokenizer: Any) -> List[List[int]]:
    """
    Tokenize hàng loạt và áp dụng Head 254 + [SEP] + Tail 254 Token Truncation.
    Tổng độ dài tối đa luôn <= 512 token (bao gồm [CLS] và [SEP]).
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    assert cls_id is not None, "Tokenizer cls_token_id không được là None"
    assert sep_id is not None, "Tokenizer sep_token_id không được là None"
    assert pad_id is not None, "Tokenizer pad_token_id không được là None"

    enc = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    all_ids = []
    for t in enc:
        if len(t) <= 510:
            ids = [cls_id] + t + [sep_id]
        else:
            ids = [cls_id] + t[:254] + [sep_id] + t[-254:] + [sep_id]
        all_ids.append(ids)
    return all_ids


class SmartBatchSampler(Sampler):
    """
    Smart Batching:
    Trộn ngẫu nhiên -> chia mega-batch (bs * 64) -> sort theo độ dài bên trong
    -> chia mini-batch -> trộn thứ tự mini-batch. Giúp dynamic padding đạt hiệu năng cao nhất.
    """
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        mega_batch_mult: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_batch_size = batch_size * mega_batch_mult
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.mega_batch_size):
            mega_chunk = indices[i:i + self.mega_batch_size]
            sorted_mega = [int(x) for x in sorted(mega_chunk, key=lambda idx: self.lengths[int(idx)])]
            for j in range(0, len(sorted_mega), self.batch_size):
                batches.append(sorted_mega[j:j + self.batch_size])

        if self.shuffle:
            batch_order = rng.permutation(len(batches))
            batches = [batches[k] for k in batch_order]

        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TextDataset(Dataset):
    """Dataset lưu trữ token ids và nhãn tương ứng."""
    def __init__(self, input_ids: List[List[int]], labels: Optional[List[int]] = None):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"input_ids": self.input_ids[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def make_collate_fn(pad_id: int):
    """Collate function thực hiện dynamic padding tới độ dài lớn nhất trong batch."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_len = max(len(x["input_ids"]) for x in batch)
        bs = len(batch)
        input_ids = torch.full((bs, batch_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, batch_len), dtype=torch.long)
        has_labels = "label" in batch[0]
        labels = []

        for i, item in enumerate(batch):
            seq = item["input_ids"]
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            if has_labels:
                labels.append(item["label"])

        res = {"input_ids": input_ids, "attention_mask": attention_mask}
        if has_labels:
            res["labels"] = torch.tensor(labels, dtype=torch.long)
        return res
    return collate_fn


class SampleWeightedLabelSmoothingLoss(nn.Module):
    """
    Hàm Loss tách biệt Label Smoothing đồng nhất và Sample-Level Class Weighting.
    Khắc phục lỗi nn.CrossEntropyLoss(weight=w, label_smoothing=eps) của PyTorch
    vốn nhân cả phần smoothing (eps/3) với weight[c] làm bơm khống xác suất PROMPT_INJECTION.
    """
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.03):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_per_sample = self.ce(logits, labels)
        sample_weights = self.class_weights[labels]
        return (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


class HybridMultiSampleClassifier(nn.Module):
    """
    Đầu phân loại Hybrid chuyên biệt cho V4:
    Kết hợp [CLS_Pooler (1152) + Masked_Mean_Pool (1152) + Masked_Max_Pool (1152)]
    đi qua LayerNorm(3456) và 5 nhánh Multi-Sample Dropout (0.1..0.5) -> Linear(3456, 3).
    Giúp bắt trọn tín hiệu độc hại nằm ẩn giữa văn bản nhập vai dài của tiếng Bồ Đào Nha (pt).
    """
    def __init__(self, hidden_size: int = 1152, num_labels: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size * 3)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.proj = nn.Linear(hidden_size * 3, num_labels)

    def forward(
        self,
        pooled_output: torch.Tensor,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(sequence_output)
        mean_pool = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        neg_val = torch.finfo(sequence_output.dtype).min / 2.0
        max_pool = sequence_output.masked_fill(attention_mask.unsqueeze(-1) == 0, neg_val).max(dim=1)[0]
        feat = torch.cat([pooled_output, mean_pool, max_pool], dim=-1)
        feat = self.norm(feat)
        logits = sum(self.proj(drop(feat)) for drop in self.dropouts) / float(len(self.dropouts))
        return logits


def forward_model_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    head_type: str = "standard",
) -> torch.Tensor:
    """Tính toán logits tương thích với cả 2 chế độ head_type ('standard' và 'hybrid_msd')."""
    if head_type == "hybrid_msd":
        base_outputs = model.rembert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = base_outputs[0]
        pooled_output = base_outputs[1]
        return model.classifier(pooled_output, sequence_output, attention_mask)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def get_optimizer_grouped_parameters(
    model: nn.Module,
    base_lr: float,
    llrd_rate: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Thiết lập Layer-wise Learning Rate Decay (LLRD) cho RemBERT (32 transformer layers):
    - rembert.embeddings.*: layer id 0
    - rembert.encoder.embedding_hidden_mapping_in.*: layer id 0 (Linear 256->1152 đặc thù RemBERT)
    - rembert.encoder.layer.{i}.*: layer id i + 1 (i = 0..31 -> 1..32)
    - rembert.pooler.*, classifier.*: layer id 33 (base_lr)
    - lr(layer_id) = base_lr * (llrd_rate ** (33 - layer_id))
    - Bias và LayerNorm.weight (bao gồm norm.weight): weight_decay = 0.0
    - Assert: tổng số params được phân nhóm == sum(p.numel() for p in model.parameters()).
    """
    no_decay_keywords = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight")
    groups: Dict[Tuple[int, float, float], List[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("rembert.embeddings.") or name.startswith("rembert.encoder.embedding_hidden_mapping_in."):
            layer_id = 0
        elif name.startswith("rembert.encoder.layer."):
            match = re.search(r"rembert\.encoder\.layer\.(\d+)\.", name)
            if not match:
                raise ValueError(f"Không trích xuất được chỉ số layer từ: {name}")
            layer_id = int(match.group(1)) + 1
            assert 1 <= layer_id <= 32, f"Layer ID ngoài phạm vi 1..32: {layer_id} ({name})"
        elif name.startswith("rembert.pooler.") or name.startswith("classifier."):
            layer_id = 33
        else:
            raise ValueError(f"Tham số không thuộc cấu trúc chuẩn của RemBERT LLRD: {name}")

        is_no_decay = any(nd in name.lower() for nd in no_decay_keywords)
        decay_val = 0.0 if is_no_decay else weight_decay
        lr_val = base_lr * (llrd_rate ** (33 - layer_id))

        group_key = (layer_id, decay_val, lr_val)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(param)

    param_groups = []
    total_grouped = 0
    for (layer_id, decay_val, lr_val), params in groups.items():
        param_groups.append({
            "params": params,
            "lr": lr_val,
            "weight_decay": decay_val,
        })
        total_grouped += sum(p.numel() for p in params)

    total_model_params = sum(p.numel() for p in model.parameters())
    assert total_grouped == total_model_params, (
        f"LLRD Param count mismatch: grouped={total_grouped} vs model={total_model_params}"
    )
    return param_groups


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    input_ids_list: List[List[int]],
    pad_id: int,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    head_type: str = "standard",
) -> np.ndarray:
    """
    Dự đoán xác suất cho tập dữ liệu:
    Sort theo độ dài để batch nhanh, sau đó KHÔI PHỤC đúng thứ tự gốc trước khi trả về.
    """
    model.eval()
    n_samples = len(input_ids_list)
    sorted_indices = sorted(range(n_samples), key=lambda i: len(input_ids_list[i]))

    sorted_probs_list = []
    for start_idx in range(0, n_samples, batch_size):
        chunk_indices = sorted_indices[start_idx:start_idx + batch_size]
        chunk_seqs = [input_ids_list[i] for i in chunk_indices]
        max_len = max(len(seq) for seq in chunk_seqs)
        bs = len(chunk_seqs)

        input_ids_cpu = torch.full((bs, max_len), pad_id, dtype=torch.long)
        attention_mask_cpu = torch.zeros((bs, max_len), dtype=torch.long)
        for row, seq in enumerate(chunk_seqs):
            seq_len = len(seq)
            input_ids_cpu[row, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask_cpu[row, :seq_len] = 1

        input_ids = input_ids_cpu.to(device, non_blocking=True)
        attention_mask = attention_mask_cpu.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = forward_model_logits(model, input_ids, attention_mask, head_type=head_type)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        sorted_probs_list.append(probs)

    sorted_probs = np.vstack(sorted_probs_list)
    orig_probs = np.zeros_like(sorted_probs, dtype=np.float32)
    orig_probs[sorted_indices] = sorted_probs
    assert orig_probs.shape == (n_samples, 3), f"Predict shape mismatch: {orig_probs.shape} vs ({n_samples}, 3)"
    return orig_probs


def compute_calibrated_val_f1(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    class_weights: List[float],
) -> Tuple[float, float, List[float]]:
    """
    Tính điểm Macro-F1 trên tập validation:
    - raw_f1: Argmax trực tiếp trên val_probs.
    - deweighted_f1: Hiệu chỉnh nhẹ độ lệch prior do class_weights gây ra (chia cho w^0.35).
    - selection_score: Trung bình 0.5 * raw_f1 + 0.5 * max(raw_f1, deweighted_f1) giúp chọn
      checkpoint có chất lượng phân lớp thực chất tốt nhất.
    """
    raw_preds = val_probs.argmax(axis=1)
    raw_f1 = float(f1_score(val_labels, raw_preds, average="macro", zero_division=0))
    per_class = f1_score(val_labels, raw_preds, average=None, zero_division=0).tolist()

    w_arr = np.array(class_weights, dtype=np.float64)
    best_cal_f1 = raw_f1
    for pwr in (0.25, 0.40, 0.55):
        adj_probs = val_probs / (w_arr ** pwr)
        adj_preds = adj_probs.argmax(axis=1)
        cand_f1 = float(f1_score(val_labels, adj_preds, average="macro", zero_division=0))
        if cand_f1 > best_cal_f1:
            best_cal_f1 = cand_f1

    selection_score = 0.5 * raw_f1 + 0.5 * best_cal_f1
    return raw_f1, selection_score, per_class


def train_single_fold(
    fold: int,
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_input_ids: List[List[int]],
    test_input_ids: List[List[int]],
    model_dir: Path,
    pad_id: int,
    device: torch.device,
    is_smoke: bool = False,
    out_dir: Path = Path("/kaggle/working"),
    log_file: Optional[Path] = None,
) -> None:
    """Huấn luyện và đánh giá trên một fold cụ thể với Intra-Fold Top-2 Snapshot Averaging."""
    head_type = config.get("HEAD_TYPE", "standard")
    top_k = int(config.get("TOP_K_CHECKPOINTS", 2))
    top1_w = float(config.get("TOP1_WEIGHT", 0.55))

    log_msg(
        f"\n{'='*70}\n>>> [BẮT ĐẦU FOLD {fold}] Head={head_type} | Top-{top_k} Snapshots "
        f"{'(SMOKE TEST)' if is_smoke else ''}\n{'='*70}",
        log_file,
    )

    fold_seed = int(config["SEED"]) + int(fold)
    set_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    oof_save_path = out_dir / f"oof_fold{fold}.npy"
    test_save_path = out_dir / f"test_fold{fold}.npy"
    val_idx_save_path = out_dir / f"val_idx_fold{fold}.npy"
    metrics_save_path = out_dir / f"metrics_fold{fold}.json"

    if not is_smoke and oof_save_path.exists() and test_save_path.exists():
        log_msg(f"[*] Fold {fold} đã có file output hoàn chỉnh từ trước. BỎ QUA (Resume)!", log_file)
        return

    val_mask = (train_df["fold"] == fold).values
    val_indices = np.where(val_mask)[0]
    train_indices = np.where(~val_mask)[0]

    if is_smoke:
        train_indices = train_indices[:256]
        val_indices = val_indices[:128]
        test_input_ids_sub = test_input_ids[:64]
    else:
        test_input_ids_sub = test_input_ids

    train_fold_ids = [train_input_ids[i] for i in train_indices]
    train_fold_labels = [int(train_df["label_id"].iloc[i]) for i in train_indices]

    val_fold_ids = [train_input_ids[i] for i in val_indices]
    val_fold_labels = np.array([int(train_df["label_id"].iloc[i]) for i in val_indices], dtype=int)

    log_msg(f"[*] Đang tải backbone RemBERT từ: {model_dir} (seed={fold_seed}, head={head_type})", log_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if head_type == "hybrid_msd":
        hidden_sz = int(getattr(model.config, "hidden_size", 1152))
        model.classifier = HybridMultiSampleClassifier(hidden_size=hidden_sz, num_labels=3)
        log_msg(f"[*] Đã kích hoạt HybridMultiSampleClassifier (3x{hidden_sz}={hidden_sz*3} -> 5x MSD -> 3)", log_file)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)

    batch_size = config["BATCH_SIZE"]
    grad_accum = config["GRAD_ACCUM"]
    collate_fn = make_collate_fn(pad_id)

    lengths = [len(seq) for seq in train_fold_ids]
    sampler = SmartBatchSampler(lengths, batch_size=batch_size, shuffle=True, seed=fold_seed)
    train_dataset = TextDataset(train_fold_ids, train_fold_labels)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    base_lr = config["LR"]
    llrd_rate = config["LLRD"]
    weight_decay = config["WEIGHT_DECAY"]
    param_groups = get_optimizer_grouped_parameters(model, base_lr, llrd_rate, weight_decay)

    optimizer = torch.optim.AdamW(param_groups, eps=config["ADAM_EPS"])

    epochs = config["EPOCHS"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = int(total_optimizer_steps * config["WARMUP_RATIO"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    class_weights_list = config["CLASS_WEIGHTS"]
    label_smoothing_val = float(config.get("LABEL_SMOOTHING", 0.03))
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float, device=device)
    criterion = SampleWeightedLabelSmoothingLoss(
        class_weights=class_weights_tensor,
        label_smoothing=label_smoothing_val,
    )

    # V5: chỉ đánh giá 1 lần tại bước cuối của epoch cuối (tiết kiệm thời gian, không snapshot)
    eval_schedule = {(epochs - 1, len(train_loader) - 1)}

    # Quản lý Top-2 checkpoints trong /tmp
    ckpt1_path = Path(f"/tmp/best1_fold{fold}.pt")
    ckpt2_path = Path(f"/tmp/best2_fold{fold}.pt")
    for p in (ckpt1_path, ckpt2_path):
        if p.exists():
            p.unlink()

    best1_score = -1.0
    best1_raw_f1 = -1.0
    best1_per_class = [0.0, 0.0, 0.0]
    best1_desc = ""

    best2_score = -1.0
    best2_raw_f1 = -1.0
    best2_desc = ""

    n_bad_steps = 0
    t0_fold = time.time()
    step_times = []

    log_msg(
        f"[*] Tổng mẫu train: {len(train_fold_ids):,} | Val: {len(val_fold_ids):,} | "
        f"Steps/epoch: {steps_per_epoch} | Total opt steps: {total_optimizer_steps}",
        log_file,
    )

    opt_step_count = 0
    stop_smoke = False

    for epoch in range(epochs):
        if stop_smoke:
            break
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        opt_step_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            is_accum_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = forward_model_logits(model, input_ids, attention_mask, head_type=head_type)
                loss = criterion(logits.float(), labels) / grad_accum

            if not torch.isfinite(loss):
                n_bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                opt_step_start = time.perf_counter()
                continue

            scaler.scale(loss).backward()

            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                optimizer.zero_grad(set_to_none=True)

                if scale_after >= scale_before:
                    scheduler.step()
                opt_step_count += 1
                step_times.append(time.perf_counter() - opt_step_start)
                opt_step_start = time.perf_counter()

            if is_smoke and opt_step_count >= 30:
                stop_smoke = True
                break

            if not is_smoke and (epoch, step) in eval_schedule:
                val_probs = predict_dataset(
                    model, val_fold_ids, pad_id, batch_size=16, device=device, head_type=head_type
                )
                raw_f1, sel_score, per_class = compute_calibrated_val_f1(
                    val_fold_labels, val_probs, class_weights_list
                )
                step_desc = f"epoch_{epoch+1}_step_{step+1}"

                log_msg(
                    f"[Eval Ep {epoch+1} Step {step+1}/{len(train_loader)}] "
                    f"Val Raw-F1: {raw_f1:.4f} (CalScore: {sel_score:.4f}) | "
                    f"BENIGN: {per_class[0]:.4f} | JAILBREAK: {per_class[1]:.4f} | PROMPT_INJECTION: {per_class[2]:.4f}",
                    log_file,
                )

                state_half = {
                    k: (v.half().cpu() if v.is_floating_point() else v.cpu())
                    for k, v in model.state_dict().items()
                }

                if sel_score > best1_score:
                    # Đẩy best1 hiện tại xuống best2 nếu đã tồn tại
                    if best1_score > 0 and top_k >= 2 and ckpt1_path.exists():
                        shutil.copy2(ckpt1_path, ckpt2_path)
                        best2_score = best1_score
                        best2_raw_f1 = best1_raw_f1
                        best2_desc = best1_desc
                    torch.save(state_half, ckpt1_path)
                    best1_score = sel_score
                    best1_raw_f1 = raw_f1
                    best1_per_class = per_class
                    best1_desc = step_desc
                elif top_k >= 2 and sel_score > best2_score:
                    torch.save(state_half, ckpt2_path)
                    best2_score = sel_score
                    best2_raw_f1 = raw_f1
                    best2_desc = step_desc

                del state_half
                model.train()
                opt_step_start = time.perf_counter()

    if is_smoke:
        smoke_val_probs = predict_dataset(
            model, val_fold_ids, pad_id, batch_size=16, device=device, head_type=head_type
        )
        smoke_test_probs = predict_dataset(
            model, test_input_ids_sub, pad_id, batch_size=16, device=device, head_type=head_type
        )
        assert smoke_val_probs.shape == (len(val_fold_ids), 3), "Smoke val shape mismatch"
        assert smoke_test_probs.shape == (len(test_input_ids_sub), 3), "Smoke test shape mismatch"

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        valid_times = step_times[1:] if len(step_times) > 1 else step_times
        sec_per_step = float(np.mean(valid_times)) if valid_times else 0.0
        full_steps_per_fold = math.ceil(11200 / (batch_size * grad_accum)) * epochs
        eta_gpu0_hours = (sec_per_step * full_steps_per_fold * 3) / 3600.0

        log_msg(f"\n{'='*70}", log_file)
        log_msg("[SMOKE TEST THÀNH CÔNG — BÁO CÁO HIỆU NĂNG]", log_file)
        log_msg(f"- Head Type: {head_type}", log_file)
        log_msg(f"- VRAM Đỉnh (peak_vram_gb): {peak_vram:.2f} GB (Ngưỡng an toàn T4 16GB: < 14.0 GB)", log_file)
        log_msg(f"- Tốc độ mỗi optimizer step (sec_per_step): {sec_per_step:.3f} giây", log_file)
        log_msg(f"- Số steps mỗi fold (full {epochs} epochs): {full_steps_per_fold}", log_file)
        log_msg(f"- ETA dự kiến cho 3 fold trên GPU 0 (ETA_hours): {eta_gpu0_hours:.2f} giờ", log_file)
        log_msg(f"- Số step lỗi NaN/Inf (n_bad_steps): {n_bad_steps}", log_file)
        log_msg(f"- Kiểm thử Eval/Test Reorder: Val {smoke_val_probs.shape} | Test {smoke_test_probs.shape} OK", log_file)
        log_msg(f"{'='*70}\n", log_file)

        del model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Tải Top-1 checkpoint và dự đoán
    log_msg(
        f"[*] Đang tải Top-1 checkpoint ({best1_desc}, Raw-F1: {best1_raw_f1:.4f}, CalScore: {best1_score:.4f})...",
        log_file,
    )
    if ckpt1_path.exists():
        sd1 = torch.load(ckpt1_path, map_location="cpu", weights_only=True)
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd1.items()})
        del sd1
    oof_probs_1 = predict_dataset(model, val_fold_ids, pad_id, batch_size=16, device=device, head_type=head_type)
    test_probs_1 = predict_dataset(model, test_input_ids_sub, pad_id, batch_size=16, device=device, head_type=head_type)

    # Nếu có Top-2 checkpoint và đạt chất lượng gần Top-1 (chênh lệch CalScore <= 0.006), kết hợp cả 2 snapshots
    if top_k >= 2 and ckpt2_path.exists() and (best1_score - best2_score) <= 0.006:
        log_msg(
            f"[*] Đang tải Top-2 checkpoint ({best2_desc}, Raw-F1: {best2_raw_f1:.4f}, CalScore: {best2_score:.4f}) "
            f"để kết hợp Intra-Fold ({top1_w:.2f} * best1 + {1.0 - top1_w:.2f} * best2)...",
            log_file,
        )
        sd2 = torch.load(ckpt2_path, map_location="cpu", weights_only=True)
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd2.items()})
        del sd2
        oof_probs_2 = predict_dataset(model, val_fold_ids, pad_id, batch_size=16, device=device, head_type=head_type)
        test_probs_2 = predict_dataset(model, test_input_ids_sub, pad_id, batch_size=16, device=device, head_type=head_type)

        oof_probs = (top1_w * oof_probs_1 + (1.0 - top1_w) * oof_probs_2).astype(np.float32)
        test_probs = (top1_w * test_probs_1 + (1.0 - top1_w) * test_probs_2).astype(np.float32)
        snap_f1 = float(f1_score(val_fold_labels, oof_probs.argmax(axis=1), average="macro", zero_division=0))
        snap_per_class = f1_score(val_fold_labels, oof_probs.argmax(axis=1), average=None, zero_division=0).tolist()
        used_step_desc = f"{best1_desc}+{best2_desc}"
        final_fold_f1 = snap_f1
        final_per_class = snap_per_class
        log_msg(
            f"[+] Intra-Fold Top-2 Snapshot F1: {snap_f1:.4f} (so với Top-1 đơn lẻ: {best1_raw_f1:.4f})",
            log_file,
        )
    else:
        oof_probs = oof_probs_1.astype(np.float32)
        test_probs = test_probs_1.astype(np.float32)
        used_step_desc = best1_desc
        final_fold_f1 = float(best1_raw_f1)
        final_per_class = best1_per_class

    assert oof_probs.shape[0] == len(val_indices), f"OOF shape mismatch: {oof_probs.shape[0]} vs {len(val_indices)}"
    assert test_probs.shape == (6000, 3), f"Test shape mismatch: got {test_probs.shape}, expected (6000, 3)"

    np.save(oof_save_path, oof_probs)
    np.save(val_idx_save_path, val_indices.astype(np.int64))
    np.save(test_save_path, test_probs)

    fold_minutes = (time.time() - t0_fold) / 60.0
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    fold_metrics = {
        "fold": fold,
        "best_f1": float(final_fold_f1),
        "best1_raw_f1": float(best1_raw_f1),
        "best2_raw_f1": float(best2_raw_f1),
        "per_class_f1": [float(x) for x in final_per_class],
        "best_epoch_step": used_step_desc,
        "n_bad_steps": int(n_bad_steps),
        "minutes": float(fold_minutes),
        "peak_vram_gb": float(peak_vram),
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)

    log_msg(
        f"[HOÀN THÀNH FOLD {fold}] Final Fold F1: {final_fold_f1:.4f} ({used_step_desc}) | "
        f"Thời gian: {fold_minutes:.1f} phút | VRAM: {peak_vram:.2f} GB",
        log_file,
    )

    for p in (ckpt1_path, ckpt2_path):
        if p.exists():
            p.unlink()
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện google/rembert 5-Fold trên Kaggle.")
    parser.add_argument("--folds", nargs="+", type=int, required=True, help="Danh sách fold cần train (vd: 0 2 4)")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working", help="Thư mục ghi kết quả")
    parser.add_argument("--smoke", action="store_true", help="Chạy chế độ Smoke Test ngắn")
    parser.add_argument("--resume_dir", type=str, default=None, help="Thư mục chứa output version cũ để resume")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_file = out_dir / "config.json"
    if not config_file.exists():
        fallback_cfg = Path("config.json")
        if fallback_cfg.exists():
            config_file = fallback_cfg
        else:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    resume_dir = args.resume_dir or config.get("RESUME_DIR")
    if resume_dir:
        res_path = Path(resume_dir)
        if res_path.exists():
            log_msg(f"[*] Sao chép checkpoint có sẵn từ resume_dir: {res_path}...")
            for f in res_path.glob("*_fold*.*"):
                dest = out_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"[*] Thiết bị tính toán: {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")

    model_dir = find_model_dir()
    log_msg(f"[*] Đã xác định model RemBERT tại: {model_dir}")
    cfg = AutoConfig.from_pretrained(str(model_dir))
    assert cfg.model_type == "rembert", f"Backbone phải là rembert, phát hiện: {cfg.model_type}"

    log_msg("[*] Đang khởi tạo tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id không được là None"

    log_msg("[*] Đang đọc train.csv, test.csv và chuẩn bị 5-Fold...")
    train_path = find_file("train.csv", required=True)
    test_path = find_file("test.csv", required=True)

    train_df = pd.read_csv(train_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""], dtype={"id": str})
    test_df = pd.read_csv(test_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""], dtype={"id": str})
    train_df = load_or_generate_folds(train_df)
    train_df["label_id"] = train_df["label"].map(LABEL2ID)

    log_msg(f"[*] Dữ liệu Train: {len(train_df):,} dòng | Test: {len(test_df):,} dòng | Folds: {sorted(train_df['fold'].unique().tolist())}")

    log_msg("[*] Chuẩn hóa văn bản và định dạng tiền tố 'lang: {language} | {clean_text}'...")
    train_texts = [
        f"lang: {row['language']} | {clean_text_safety(row['text'])}"
        for _, row in train_df.iterrows()
    ]
    test_texts = [
        f"lang: {row['language']} | {clean_text_safety(row['text'])}"
        for _, row in test_df.iterrows()
    ]

    log_msg("[*] Tokenize trước toàn bộ tập Train và Test (Head 254 + [SEP] + Tail 254)...")
    train_input_ids = prepare_input_ids(train_texts, tokenizer)
    test_input_ids = prepare_input_ids(test_texts, tokenizer)

    for fold in args.folds:
        fold_log_file = logs_dir / f"fold{fold}.log"
        train_single_fold(
            fold=fold,
            config=config,
            train_df=train_df,
            test_df=test_df,
            train_input_ids=train_input_ids,
            test_input_ids=test_input_ids,
            model_dir=model_dir,
            pad_id=pad_id,
            device=device,
            is_smoke=args.smoke,
            out_dir=out_dir,
            log_file=fold_log_file,
        )


if __name__ == "__main__":
    main()
