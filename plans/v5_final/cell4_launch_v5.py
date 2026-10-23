# =============================================================================
# 🚀 4. (V5) KHỞI CHẠY 2 GPU SONG SONG + CHỐT CHẶN THỜI GIAN (HARD DEADLINE)
# =============================================================================
# GPU 0: fold 0 -> 2 | GPU 1: fold 1 -> 3. Quá DEADLINE_UTC thì kill process,
# KHÔNG raise lỗi -> Cell 5 vẫn ghép submission từ các fold đã xong.
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch

# 17:40 giờ Việt Nam (UTC+7) = 10:40 UTC. Kaggle chạy giờ UTC.
DEADLINE_UTC = datetime(2026, 10, 22, 10, 40, 0, tzinfo=timezone.utc)

if SMOKE_TEST:
    print("[!] SMOKE_TEST = True: Bỏ qua cell khởi chạy chính thức.", flush=True)
else:
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[*] Phát hiện {n_gpus} GPU | Deadline UTC: {DEADLINE_UTC.isoformat()} | Now UTC: {datetime.now(timezone.utc).isoformat()}", flush=True)
    log_dir = Path(OUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = GPU_FOLDS if n_gpus >= 2 else {"0": sorted({f for fl in GPU_FOLDS.values() for f in fl})}
    processes, log_files, log_paths = [], [], []
    for gpu_id, fold_list in plan.items():
        if not fold_list:
            continue
        log_path = log_dir / f"gpu{gpu_id}_launch.log"
        lf = open(log_path, "w", encoding="utf-8", buffering=1)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [sys.executable, "train_rembert.py", "--folds", *[str(f) for f in fold_list], "--out_dir", OUT_DIR]
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        processes.append(p); log_files.append(lf); log_paths.append(log_path)
        print(f"[+] GPU {gpu_id} (PID {p.pid}) -> folds {fold_list}", flush=True)

    start_time = time.time()
    killed = False
    while any(p.poll() is None for p in processes):
        time.sleep(60)
        now = datetime.now(timezone.utc)
        print(f"\n--- [{(time.time() - start_time) / 60:.1f} phút | UTC {now.strftime('%H:%M:%S')}] ---", flush=True)
        for i, (proc, lp) in enumerate(zip(processes, log_paths)):
            st = "Đang chạy" if proc.poll() is None else f"Xong (mã {proc.poll()})"
            tail = lp.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:] if lp.exists() else []
            print(f"> P{i} {st}\n  " + "\n  ".join(tail), flush=True)
        if now >= DEADLINE_UTC:
            print("[!!!] CHẠM DEADLINE -> KILL các process còn chạy, chuyển sang ghép submission từ fold đã xong.", flush=True)
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
            killed = True
            break

    for lf in log_files:
        lf.close()
    for i, (proc, lp) in enumerate(zip(processes, log_paths)):
        rc = proc.poll()
        if rc not in (0, None) and not killed:
            tail = "\n".join(lp.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-40:])
            print(f"[WARN] Process {i} lỗi mã {rc} (KHÔNG raise, Cell 5 dùng fold đã xong):\n{tail}", flush=True)
    done = [k for k in range(5) if (Path(OUT_DIR) / f"test_fold{k}.npy").exists()]
    print(f"[+] Kết thúc huấn luyện. Fold hoàn thành: {done}", flush=True)
