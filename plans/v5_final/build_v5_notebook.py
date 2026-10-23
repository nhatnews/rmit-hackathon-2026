"""Build notebooks/rembert_v5_kaggle.ipynb from the proven V3 notebook with minimal, verified diffs."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(r"D:\Code\rmit-hackathon-2026")
SRC = ROOT / "notebooks" / "rembert_v3_kaggle.ipynb"
DST = ROOT / "notebooks" / "rembert_v5_kaggle.ipynb"
PLAN = ROOT / "plans" / "v5_final"

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def get(i):
    return "".join(cells[i]["source"])


def put(i, s):
    cells[i]["source"] = s.splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


def sub1(s, old, new):
    assert s.count(old) == 1, f"pattern count != 1: {old!r} -> {s.count(old)}"
    return s.replace(old, new)


# ---- Cell 0: markdown header (line 1 must stay '# Model: google/rembert')
c0 = get(0)
assert c0.startswith("# Model: google/rembert")
c0 = c0.replace("(V3)", "(V5 — Final Shot 6/6)")
put(0, c0)

# ---- Cell 1: config
c1 = get(1)
c1 = sub1(c1, 'VERSION_TAG = "V3"', 'VERSION_TAG = "V5"')
c1 = sub1(c1, "EPOCHS = 4 ", "EPOCHS = 3 ")
c1 = sub1(c1, "CLASS_WEIGHTS = [1.0, 1.5, 5.0]", "CLASS_WEIGHTS = [1.0, 1.8, 7.0]   # V5: giữa V2 [1,2,8.8] và V3 [1,1.5,5.0] -> lấy lại en-PI")
c1 = sub1(c1, "TOP_K_CHECKPOINTS = 2 ", "TOP_K_CHECKPOINTS = 1 ")
c1 = sub1(c1, 'GPU_FOLDS = {"0": [0, 2, 4], "1": [1, 3]}', 'GPU_FOLDS = {"0": [0, 2], "1": [1, 3]}   # V5: 4 fold (0-3), 2+2 để kịp deadline')
put(1, c1)

# ---- Cell 2: train_rembert.py
c2 = get(2)
# (a) fold key lowercase -> khớp 100% data/processed/folds.csv (đã kiểm chứng local: match=1.0)
c2 = sub1(
    c2,
    'lambda s: re.sub(r"\\s+", " ", clean_text_safety(s))\n',
    'lambda s: re.sub(r"\\s+", " ", clean_text_safety(s).lower())  # V5: .lower() khớp folds.csv\n',
)
# (b) chỉ đánh giá 1 lần ở cuối epoch cuối
old_sched = c2[c2.index("    eval_schedule = set()\n"):c2.index("    # Quản lý Top-2 checkpoints trong /tmp\n")]
new_sched = (
    "    # V5: chỉ đánh giá 1 lần tại bước cuối của epoch cuối (tiết kiệm thời gian, không snapshot)\n"
    "    eval_schedule = {(epochs - 1, len(train_loader) - 1)}\n\n"
)
c2 = c2.replace(old_sched, new_sched)
put(2, c2)

# ---- Cell 4 & 5: replace with tested V5 code
put(4, (PLAN / "cell4_launch_v5.py").read_text(encoding="utf-8"))
put(5, (PLAN / "cell5_aggregate_v5.py").read_text(encoding="utf-8"))

# ---- syntax check every code cell (strip %%writefile magic)
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    s = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith(("%%", "!", "%")))
    ast.parse(s)
    c["outputs"] = []
    c["execution_count"] = None

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("[OK] wrote", DST)
