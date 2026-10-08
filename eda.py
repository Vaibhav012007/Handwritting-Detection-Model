"""
eda.py — Exploratory Data Analysis
Produces:
  results/eda_class_balance.png   — bar chart per class
  results/eda_size_distribution.png — image W×H scatter
  results/eda_sample_grid.png     — 6×6 sample grid
  Console output: duplicate/leakage report
"""

import os
import sys
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import imagehash
from collections import Counter

# ── Bootstrap path so common.py is importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import set_seed, DATA_ROOT, RESULTS_DIR

set_seed()
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_csv(split: str):
    csv_path = os.path.join(DATA_ROOT, split, "labels.csv")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def image_path(split: str, img_name: str) -> str:
    return os.path.join(DATA_ROOT, split, "images", img_name)


# ─── Load all splits ─────────────────────────────────────────────────────────
print("Loading metadata...")
train_rows = load_csv("train")
val_rows = load_csv("val")
test_rows = load_csv("test")

print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

# ─── Class balance ────────────────────────────────────────────────────────────
train_brands = Counter(r["MEDICINE_NAME"] for r in train_rows)
val_brands   = Counter(r["MEDICINE_NAME"] for r in val_rows)
test_brands  = Counter(r["MEDICINE_NAME"] for r in test_rows)

all_brands = sorted(train_brands.keys())
n_classes = len(all_brands)
print(f"\nUnique brands: {n_classes}")

# Check balance
brand_counts = [train_brands[b] for b in all_brands]
print(f"Samples per class (train): min={min(brand_counts)} max={max(brand_counts)} mean={np.mean(brand_counts):.1f}")

# Plot
fig, ax = plt.subplots(figsize=(18, 5))
x = np.arange(n_classes)
y_train = np.array([train_brands[b] for b in all_brands], dtype=np.float64)
y_val = np.array([val_brands.get(b, 0) for b in all_brands], dtype=np.float64)
y_test = np.array([test_brands.get(b, 0) for b in all_brands], dtype=np.float64)

ax.bar(x, y_train, label="train", alpha=0.8)
ax.bar(x, y_val, bottom=y_train, label="val", alpha=0.8)
ax.bar(x, y_test, bottom=y_train + y_val, label="test", alpha=0.8)

ax.set_xticks(list(x))
ax.set_xticklabels(all_brands, rotation=90, fontsize=6)
ax.set_ylabel("Count")
ax.set_title(f"Class Balance — {n_classes} brands (perfectly balanced {brand_counts[0]} train / {list(val_brands.values())[0]} val / {list(test_brands.values())[0]} test per class)")
ax.legend()
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "eda_class_balance.png")
plt.savefig(out_path, dpi=120)
plt.close()
print(f"Saved: {out_path}")

# ─── Image size distribution ──────────────────────────────────────────────────
print("\nSampling image sizes (training set)...")
widths, heights = [], []
for row in train_rows[:500]:  # sample 500
    p = image_path("train", row["IMAGE"])
    if os.path.exists(p):
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

print(f"Width:  min={min(widths)} max={max(widths)} mean={np.mean(widths):.1f}")
print(f"Height: min={min(heights)} max={max(heights)} mean={np.mean(heights):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(widths, bins=30, color="#4a90d9")
axes[0].set_title("Image Width distribution"); axes[0].set_xlabel("Pixels")
axes[1].hist(heights, bins=30, color="#e85d5d")
axes[1].set_title("Image Height distribution"); axes[1].set_xlabel("Pixels")
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "eda_size_distribution.png")
plt.savefig(out_path, dpi=120)
plt.close()
print(f"Saved: {out_path}")

# ─── 6×6 Sample grid ─────────────────────────────────────────────────────────
print("\nBuilding sample grid...")
selected = random.sample(train_rows, min(36, len(train_rows)))
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(6, 6, figure=fig, hspace=0.5, wspace=0.3)
for i, row in enumerate(selected):
    ax = fig.add_subplot(gs[i // 6, i % 6])
    p = image_path("train", row["IMAGE"])
    with Image.open(p) as img:
        img_arr = np.asarray(img)
        ax.imshow(img_arr, cmap="gray", aspect="auto")
    ax.set_title(row["MEDICINE_NAME"], fontsize=5, pad=2)
    ax.axis("off")
plt.suptitle("Training Sample Grid (36 random crops)", fontsize=12, y=1.01)
out_path = os.path.join(RESULTS_DIR, "eda_sample_grid.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ─── Perceptual hash deduplication / leakage check ───────────────────────────
print("\nComputing perceptual hashes for leakage detection...")
HASH_SIZE = 8
HAMMING_THRESH = 4

def compute_hashes(rows, split):
    hashes = {}
    for row in rows:
        p = image_path(split, row["IMAGE"])
        if os.path.exists(p):
            with Image.open(p) as img:
                h = imagehash.phash(img, hash_size=HASH_SIZE)
            hashes[(split, row["IMAGE"])] = h
    return hashes

train_hashes = compute_hashes(train_rows, "train")
val_hashes   = compute_hashes(val_rows,   "val")
test_hashes  = compute_hashes(test_rows,  "test")

print(f"  Hashed: {len(train_hashes)} train, {len(val_hashes)} val, {len(test_hashes)} test")

# Check train vs val leakage
leakage_tv, leakage_tt = 0, 0
SAMPLE_LIMIT = 3000  # avoid O(n^2) explosion — sample train
sample_train_keys = list(train_hashes.keys())[:SAMPLE_LIMIT]

for tk in sample_train_keys:
    th = train_hashes[tk]
    for vk, vh in val_hashes.items():
        if th - vh <= HAMMING_THRESH:
            leakage_tv += 1
            break
    for xk, xh in test_hashes.items():
        if th - xh <= HAMMING_THRESH:
            leakage_tt += 1
            break

print(f"\nLeakage report (hamming <= {HAMMING_THRESH}):")
print(f"  Train<->Val near-duplicates : {leakage_tv}")
print(f"  Train<->Test near-duplicates: {leakage_tt}")
if leakage_tv + leakage_tt == 0:
    print("  [OK] No detectable train/test or train/val leakage found.")
else:
    print("  [WARNING] Potential leakage detected — see counts above.")

# ─── Generic distribution ─────────────────────────────────────────────────────
generic_counts = Counter(r["GENERIC_NAME"] for r in train_rows)
print(f"\nGeneric classes: {len(generic_counts)}")
for g, cnt in sorted(generic_counts.items(), key=lambda x: -x[1]):
    print(f"  {g:<30s} {cnt}")

print("\nEDA complete.")
