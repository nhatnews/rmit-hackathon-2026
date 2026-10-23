"""
train_rembert.py
Script huấn luyện 5-Fold chuyên biệt cho google/rembert trên Kaggle GPU T4.
Tuân thủ nghiêm ngặt luật Single-Model (K-Fold checkpoint averaging của DUY NHẤT RemBERT).
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


def find_file(name: str) -> Path:
    """Tìm file dữ liệu trong /kaggle/input hoặc các thư mục dữ liệu cục bộ."""
    search_dirs = [Path("/kaggle/input"), Path("data"), Path(".")]
    for base in search_dirs:
        if base.exists():
            for f in base.rglob(name):
                if f.is_file():
                    return f
    raise FileNotFoundError(f"Không tìm thấy tệp bắt buộc: '{name}'. Vui lòng kiểm tra lại Input!")


def clean_text_safety(text: Any) -> str:
    """
    Chuẩn hóa văn bản an toàn cho tấn công đa ngôn ngữ (pt, fr, it, en).
    BẢO TỒN CHỮ HOA/THƯỜNG (preserve_case=True) và DẤU ROMANCE.
    """
    if not isinstance(text, str):
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
    # Chuẩn hóa Unicode NFC (giữ nguyên các ký tự có dấu Romance)
    text = unicodedata.normalize("NFC", text)
    # Loại bỏ null byte có thể gây lỗi parser
    text = text.replace("\x00", "")
    # Chuẩn hóa xuống dòng
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def prepare_input_ids(texts: List[str], tokenizer: Any) -> List[List[int]]:
    """
    Tokenize hàng loạt và áp dụng Head 254 + [SEP] + Tail 254 Token Truncation.
    Token [SEP] ở giữa thông báo cho Transformer biết đây là hai phân đoạn tách biệt
    (Opening Preamble vs Closing Payload), loại bỏ bigram giả ở mối nối.
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
    - Bias và LayerNorm.weight: weight_decay = 0.0
    - Assert: tổng số params được phân nhóm == sum(p.numel() for p in model.parameters()).
    """
    no_decay_keywords = ("bias", "layernorm.weight", "layer_norm.weight")
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

        # Cấp phát trên CPU rồi chuyển cả batch lên GPU 1 lần duy nhất
        input_ids_cpu = torch.full((bs, max_len), pad_id, dtype=torch.long)
        attention_mask_cpu = torch.zeros((bs, max_len), dtype=torch.long)
        for row, seq in enumerate(chunk_seqs):
            seq_len = len(seq)
            input_ids_cpu[row, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask_cpu[row, :seq_len] = 1

        input_ids = input_ids_cpu.to(device, non_blocking=True)
        attention_mask = attention_mask_cpu.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        sorted_probs_list.append(probs)

    sorted_probs = np.vstack(sorted_probs_list)
    orig_probs = np.zeros_like(sorted_probs, dtype=np.float32)
    orig_probs[sorted_indices] = sorted_probs
    assert orig_probs.shape == (n_samples, 3), f"Predict shape mismatch: {orig_probs.shape} vs ({n_samples}, 3)"
    return orig_probs


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
    """Huấn luyện và đánh giá trên một fold cụ thể."""
    log_msg(f"\n{'='*70}\n>>> [BẮT ĐẦU FOLD {fold}] {'(SMOKE TEST)' if is_smoke else ''}\n{'='*70}", log_file)

    # Cố định seed + fold (Đặc tả C10) và reset bộ đếm VRAM đỉnh
    fold_seed = int(config["SEED"]) + int(fold)
    set_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Kiểm tra resume nếu fold đã hoàn thành
    oof_save_path = out_dir / f"oof_fold{fold}.npy"
    test_save_path = out_dir / f"test_fold{fold}.npy"
    val_idx_save_path = out_dir / f"val_idx_fold{fold}.npy"
    metrics_save_path = out_dir / f"metrics_fold{fold}.json"

    if not is_smoke and oof_save_path.exists() and test_save_path.exists():
        log_msg(f"[*] Fold {fold} đã có file output hoàn chỉnh từ trước. BỎ QUA (Resume)!", log_file)
        return

    # Tách train và val theo fold (val_indices luôn theo thứ tự tăng dần)
    val_mask = (train_df["fold"] == fold).values
    val_indices = np.where(val_mask)[0]
    train_indices = np.where(~val_mask)[0]

    if is_smoke:
        # Cắt mẫu nhỏ cho smoke test theo đúng đặc tả B: 256 train, 128 val, 64 test
        train_indices = train_indices[:256]
        val_indices = val_indices[:128]
        test_input_ids_sub = test_input_ids[:64]
    else:
        test_input_ids_sub = test_input_ids

    train_fold_ids = [train_input_ids[i] for i in train_indices]
    train_fold_labels = [int(train_df["label_id"].iloc[i]) for i in train_indices]

    val_fold_ids = [train_input_ids[i] for i in val_indices]
    val_fold_labels = np.array([int(train_df["label_id"].iloc[i]) for i in val_indices], dtype=int)

    # Khởi tạo model từ checkpoint RemBERT
    log_msg(f"[*] Đang tải backbone RemBERT từ: {model_dir} (seed={fold_seed})", log_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Kích hoạt Gradient Checkpointing để tiết kiệm VRAM trên T4
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)

    # DataLoader với SmartBatchSampler
    batch_size = config["BATCH_SIZE"]
    grad_accum = config["GRAD_ACCUM"]
    collate_fn = make_collate_fn(pad_id)

    lengths = [len(seq) for seq in train_fold_ids]
    sampler = SmartBatchSampler(lengths, batch_size=batch_size, shuffle=True, seed=fold_seed)
    train_dataset = TextDataset(train_fold_ids, train_fold_labels)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    # Tối ưu hóa & LLRD
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
    class_weights_tensor = torch.tensor(config["CLASS_WEIGHTS"], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)

    # Điểm đánh giá (eval 2 lần/epoch từ epoch 2, và cuối epoch 1; căn chỉnh đúng biên grad_accum)
    eval_schedule = set()
    for ep in range(epochs):
        if ep == 0:
            eval_schedule.add((ep, len(train_loader) - 1))
        else:
            half_opt = max(1, steps_per_epoch // 2)
            mid_pt = min(len(train_loader) - 1, half_opt * grad_accum - 1)
            eval_schedule.add((ep, mid_pt))
            eval_schedule.add((ep, len(train_loader) - 1))

    # Quản lý checkpoint tốt nhất (lưu trong /tmp để không chiếm output Kaggle)
    ckpt_path = Path(f"/tmp/best_fold{fold}.pt")
    if ckpt_path.exists():
        ckpt_path.unlink()

    best_f1 = -1.0
    best_per_class = [0.0, 0.0, 0.0]
    best_step_desc = ""
    n_bad_steps = 0
    t0_fold = time.time()
    step_times = []

    log_msg(
        f"[*] Tổng mẫu train: {len(train_fold_ids):,} | Val: {len(val_fold_ids):,} | "
        f"Steps/epoch: {steps_per_epoch} | Total opt steps: {total_optimizer_steps}",
        log_file,
    )

    # Vòng lặp huấn luyện
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
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

            # Đánh giá trong chế độ SMOKE_TEST
            if is_smoke and opt_step_count >= 30:
                stop_smoke = True
                break

            # Đánh giá validation theo lịch trình
            if not is_smoke and (epoch, step) in eval_schedule:
                val_probs = predict_dataset(model, val_fold_ids, pad_id, batch_size=16, device=device)
                val_preds = val_probs.argmax(axis=1)
                macro_f1 = float(f1_score(val_fold_labels, val_preds, average="macro", zero_division=0))
                per_class = f1_score(val_fold_labels, val_preds, average=None, zero_division=0).tolist()

                log_msg(
                    f"[Eval Ep {epoch+1} Step {step+1}/{len(train_loader)}] "
                    f"Val Macro-F1: {macro_f1:.4f} | "
                    f"BENIGN: {per_class[0]:.4f} | JAILBREAK: {per_class[1]:.4f} | PROMPT_INJECTION: {per_class[2]:.4f}",
                    log_file,
                )

                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_per_class = per_class
                    best_step_desc = f"epoch_{epoch+1}_step_{step+1}"
                    # Lưu checkpoint nửa chính xác (FP16) trên CPU vào /tmp
                    torch.save(
                        {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()},
                        ckpt_path,
                    )

                model.train()
                opt_step_start = time.perf_counter()

    # Báo cáo kết quả Smoke Test (bao gồm kiểm thử predict_dataset trên 128 val + 64 test)
    if is_smoke:
        smoke_val_probs = predict_dataset(model, val_fold_ids, pad_id, batch_size=16, device=device)
        smoke_test_probs = predict_dataset(model, test_input_ids_sub, pad_id, batch_size=16, device=device)
        assert smoke_val_probs.shape == (len(val_fold_ids), 3), "Smoke val shape mismatch"
        assert smoke_test_probs.shape == (len(test_input_ids_sub), 3), "Smoke test shape mismatch"

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        # Bỏ bước đầu tiên khi tính trung bình tốc độ để tránh chi phí khởi tạo CUDA kernel
        valid_times = step_times[1:] if len(step_times) > 1 else step_times
        sec_per_step = float(np.mean(valid_times)) if valid_times else 0.0
        full_steps_per_fold = math.ceil(11200 / (batch_size * grad_accum)) * epochs
        eta_gpu0_hours = (sec_per_step * full_steps_per_fold * 3) / 3600.0

        log_msg(f"\n{'='*70}", log_file)
        log_msg("[SMOKE TEST THÀNH CÔNG — BÁO CÁO HIỆU NĂNG]", log_file)
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

    # Tải lại checkpoint tốt nhất và dự đoán cho full fold
    log_msg(f"[*] Đang tải checkpoint tốt nhất ({best_step_desc}, Macro-F1: {best_f1:.4f}) từ {ckpt_path}...", log_file)
    if ckpt_path.exists():
        state_dict_half = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict_float = {k: (v.float() if v.is_floating_point() else v) for k, v in state_dict_half.items()}
        model.load_state_dict(state_dict_float)
    else:
        log_msg("[!] Cảnh báo: Không tìm thấy checkpoint /tmp, sử dụng trọng số hiện tại.", log_file)

    # Dự đoán Out-Of-Fold và Test (Đặc tả C6)
    log_msg(f"[*] Đang sinh OOF probabilities cho Fold {fold}...", log_file)
    oof_probs = predict_dataset(model, val_fold_ids, pad_id, batch_size=16, device=device)
    assert oof_probs.shape[0] == len(val_indices), f"OOF shape mismatch: {oof_probs.shape[0]} vs {len(val_indices)}"

    log_msg(f"[*] Đang sinh Test probabilities cho Fold {fold}...", log_file)
    test_probs = predict_dataset(model, test_input_ids_sub, pad_id, batch_size=16, device=device)
    assert test_probs.shape == (6000, 3), f"Test shape mismatch: got {test_probs.shape}, expected (6000, 3)"

    # Lưu kết quả fold
    np.save(oof_save_path, oof_probs.astype(np.float32))
    np.save(val_idx_save_path, val_indices.astype(np.int64))
    np.save(test_save_path, test_probs.astype(np.float32))

    fold_minutes = (time.time() - t0_fold) / 60.0
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    fold_metrics = {
        "fold": fold,
        "best_f1": float(best_f1),
        "per_class_f1": [float(x) for x in best_per_class],
        "best_epoch_step": best_step_desc,
        "n_bad_steps": int(n_bad_steps),
        "minutes": float(fold_minutes),
        "peak_vram_gb": float(peak_vram),
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)

    log_msg(
        f"[HOÀN THÀNH FOLD {fold}] Best F1: {best_f1:.4f} | "
        f"Thời gian: {fold_minutes:.1f} phút | VRAM: {peak_vram:.2f} GB",
        log_file,
    )

    # Dọn dẹp checkpoint và bộ nhớ GPU
    if ckpt_path.exists():
        ckpt_path.unlink()
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

    # Đọc cấu hình từ config.json
    config_file = out_dir / "config.json"
    if not config_file.exists():
        fallback_cfg = Path("config.json")
        if fallback_cfg.exists():
            config_file = fallback_cfg
        else:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resume: copy file có sẵn từ resume_dir (ưu tiên CLI arg, sau đó tới config.json)
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

    # Tìm backbone và kiểm tra đúng loại RemBERT
    model_dir = find_model_dir()
    log_msg(f"[*] Đã xác định model RemBERT tại: {model_dir}")
    cfg = AutoConfig.from_pretrained(str(model_dir))
    assert cfg.model_type == "rembert", f"Backbone phải là rembert, phát hiện: {cfg.model_type}"

    # Đọc tokenizer
    log_msg("[*] Đang khởi tạo tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id không được là None"

    # Đọc dữ liệu (keep_default_na=False để các chuỗi như 'null'/'NA' không bị biến thành NaN)
    log_msg("[*] Đang đọc train.csv, test.csv và folds.csv...")
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    folds_path = find_file("folds.csv")

    train_df = pd.read_csv(train_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""], dtype={"id": str})
    test_df = pd.read_csv(test_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""], dtype={"id": str})
    folds_df = pd.read_csv(folds_path, dtype={"id": str})

    # Merge folds nghiêm ngặt theo ID (validate="1:1"), TUYỆT ĐỐI KHÔNG FALLBACK ÂM THẦM
    train_df = train_df.merge(folds_df, on="id", how="left", validate="1:1")
    assert train_df["fold"].notna().all(), (
        f"LỖI NGHIÊM TRỌNG: folds.csv không khớp với train.csv! Có {train_df['fold'].isna().sum()} dòng NaN."
    )
    train_df["fold"] = train_df["fold"].astype(int)
    train_df["label_id"] = train_df["label"].map(LABEL2ID)

    log_msg(f"[*] Dữ liệu Train: {len(train_df):,} dòng | Test: {len(test_df):,} dòng | Folds: {sorted(train_df['fold'].unique().tolist())}")

    # Chuẩn hóa văn bản và format tiền tố ngôn ngữ
    log_msg("[*] Chuẩn hóa văn bản và định dạng tiền tố 'lang: {language} | {clean_text}'...")
    train_texts = [
        f"lang: {row['language']} | {clean_text_safety(row['text'])}"
        for _, row in train_df.iterrows()
    ]
    test_texts = [
        f"lang: {row['language']} | {clean_text_safety(row['text'])}"
        for _, row in test_df.iterrows()
    ]

    # Tokenize và cắt Head 254 + [SEP] + Tail 254
    log_msg("[*] Tokenize trước toàn bộ tập Train và Test (Head 254 + [SEP] + Tail 254)...")
    train_input_ids = prepare_input_ids(train_texts, tokenizer)
    test_input_ids = prepare_input_ids(test_texts, tokenizer)

    # Chạy lần lượt từng fold được phân công cho process này
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
