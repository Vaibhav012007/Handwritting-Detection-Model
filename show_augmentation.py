"""
show_augmentation.py — Visualise training augmentation.
Saves results/augmentation_grid.png showing 4 originals × 4 augmented variants.
"""

import os, sys, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import set_seed, DATA_ROOT, RESULTS_DIR, load_split, TRAIN_AUGMENT, preprocess_image

set_seed(42)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading training dataset...")
train_ds = load_split("train", augment=False)

# Pick 4 diverse classes
all_items = train_ds.items
by_brand = {}
for item in all_items:
    b = item[1]
    if b not in by_brand:
        by_brand[b] = item

originals = random.sample(list(by_brand.values()), 4)

VARIANTS = 4
fig, axes = plt.subplots(4, 1 + VARIANTS, figsize=(14, 10))

for row_idx, (img_path, brand, generic) in enumerate(originals):
    # Original preprocessed
    orig = preprocess_image(img_path)
    axes[row_idx][0].imshow(np.array(orig), cmap="gray", aspect="auto")
    axes[row_idx][0].set_title(f"{brand}\n(preprocessed)", fontsize=7)
    axes[row_idx][0].axis("off")

    # Augmented variants
    for col_idx in range(1, 1 + VARIANTS):
        aug = TRAIN_AUGMENT(orig)
        axes[row_idx][col_idx].imshow(np.array(aug), cmap="gray", aspect="auto")
        axes[row_idx][col_idx].set_title(f"Aug {col_idx}", fontsize=7)
        axes[row_idx][col_idx].axis("off")

plt.suptitle("Augmentation Grid: 4 classes × (original + 4 augmented variants)", fontsize=11)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "augmentation_grid.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
