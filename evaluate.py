"""
evaluate.py — Full evaluation on the test split.

Steps:
  1. Tune acceptance threshold on VALIDATION set (threshold is never touched by test).
  2. Run full pipeline on TEST split (called once).
  3. Write results/metrics.json.
  4. Save sample prediction images (correct / incorrect / uncertain).
  5. Run ablation table.

Never modifies train or val in any way after threshold tuning.
"""

import os, sys, json, csv, math, shutil
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    set_seed, SEED, DEVICE, DATA_ROOT, RESULTS_DIR, CHECKPOINTS_DIR,
    load_split, load_brand_generic_map, preprocess_image
)
from matcher import (
    MedicineMatcher, build_medicine_dict, tune_threshold, run_ablation
)
from pipeline import run_trocr_robust, write_record

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EVALUATION SCRIPT")
print("=" * 60)

# ─── Load model ───────────────────────────────────────────────────────────────
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
best_ckpt = os.path.join(CHECKPOINTS_DIR, "trocr_best")
if not os.path.exists(best_ckpt):
    raise FileNotFoundError(f"No checkpoint at {best_ckpt}. Run train_trocr.py first.")

print(f"Loading model from {best_ckpt}...")
processor = TrOCRProcessor.from_pretrained(best_ckpt)
model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(DEVICE)
model.eval()

# ─── Brand → generic map ──────────────────────────────────────────────────────
brand_to_generic = load_brand_generic_map()

# ─── Helper: run TrOCR on a DataLoader ────────────────────────────────────────
MAX_TARGET_LENGTH = 32
BATCH_SIZE = 8


def collate_fn(batch):
    images, med_names, gen_names = zip(*batch)
    encoding = processor(images=list(images), return_tensors="pt", padding=True)
    return encoding.pixel_values, list(med_names), list(gen_names)


import Levenshtein as lev_lib

def char_error_rate(preds, gts):
    errors, total = 0, 0
    for p, g in zip(preds, gts):
        errors += lev_lib.distance(p, g)
        total += max(len(g), 1)
    return errors / total if total > 0 else 0.0


def infer_split(loader, desc="Infer"):
    """Return (preds, gt_brands, gt_generics, ocr_confs)."""
    all_preds, all_gt_brands, all_gt_generics, all_confs = [], [], [], []
    with torch.no_grad():
        for pixel_values, med_names, gen_names in tqdm(loader, desc=desc):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model.generate(
                pixel_values,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_preds = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            batch_preds = [p.strip() for p in batch_preds]

            # Confidence: mean of max softmax prob per step
            for i in range(len(batch_preds)):
                try:
                    if outputs.scores:
                        max_probs = [
                            torch.softmax(step_scores[i], dim=-1).max().item()
                            for step_scores in outputs.scores
                        ]
                        conf = float(sum(max_probs) / max(len(max_probs), 1))
                    else:
                        conf = 1.0
                except Exception:
                    conf = 1.0
                all_confs.append(conf)

            all_preds.extend(batch_preds)
            all_gt_brands.extend(med_names)
            all_gt_generics.extend(gen_names)

    return all_preds, all_gt_brands, all_gt_generics, all_confs


# ─── 1. Run on validation set for threshold tuning ───────────────────────────
print("\n[1/3] Running inference on VALIDATION set (for threshold tuning)...")
val_ds = load_split("val", augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
val_preds, val_gt_brands, val_gt_generics, val_confs = infer_split(val_loader, "Val inference")

val_cer = char_error_rate(val_preds, val_gt_brands)
val_brand_acc = sum(p == g for p, g in zip(val_preds, val_gt_brands)) / len(val_preds)
val_generic_acc = sum(
    brand_to_generic.get(p, "?") == brand_to_generic.get(g, "!")
    for p, g in zip(val_preds, val_gt_brands)
) / len(val_preds)

print(f"Val CER          : {val_cer:.4f}")
print(f"Val brand acc    : {val_brand_acc:.4f}")
print(f"Val generic acc  : {val_generic_acc:.4f}")

print("\nTuning acceptance threshold on VALIDATION set...")
best_threshold = tune_threshold(val_preds, val_confs, val_gt_brands, brand_to_generic)

# Save threshold
thresh_path = os.path.join(RESULTS_DIR, "best_threshold.json")
with open(thresh_path, "w") as f:
    json.dump({"threshold": best_threshold}, f, indent=2)
print(f"Threshold saved: {thresh_path}")

# ─── 2. Run on test set (ONCE) ────────────────────────────────────────────────
print("\n[2/3] Running inference on TEST set (called once, never again for tuning)...")
test_ds = load_split("test", augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)
test_preds, test_gt_brands, test_gt_generics, test_confs = infer_split(test_loader, "Test inference")

test_cer = char_error_rate(test_preds, test_gt_brands)
test_brand_acc = sum(p == g for p, g in zip(test_preds, test_gt_brands)) / len(test_preds)
test_generic_acc = sum(
    brand_to_generic.get(p, "?") == brand_to_generic.get(g, "!")
    for p, g in zip(test_preds, test_gt_brands)
) / len(test_preds)

print(f"Test CER          : {test_cer:.4f}")
print(f"Test brand acc    : {test_brand_acc:.4f}")
print(f"Test generic acc  : {test_generic_acc:.4f}")

# ─── Ablation + threshold curve ───────────────────────────────────────────────
print("\n[3/3] Running ablation analysis...")
ablation = run_ablation(
    test_preds, test_confs, test_gt_brands, test_gt_generics,
    brand_to_generic, best_threshold
)

# ─── Full metrics.json ────────────────────────────────────────────────────────
metrics = {
    "model": "microsoft/trocr-small-handwritten (fine-tuned)",
    "device": "CPU" if DEVICE.type == "cpu" else f"GPU:{torch.cuda.get_device_name(0)}",
    "dataset": {
        "train": 3120,
        "val": 780,
        "test": 780,
        "n_brand_classes": 78,
        "n_generic_classes": 15,
    },
    "validation": {
        "cer": val_cer,
        "brand_acc": val_brand_acc,
        "generic_acc": val_generic_acc,
    },
    "test": {
        "cer": test_cer,
        "brand_acc": test_brand_acc,
        "generic_acc": test_generic_acc,
    },
    "confidence_threshold": best_threshold,
    "ablation": ablation,
}

metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved: {metrics_path}")

# ─── Sample predictions grid ──────────────────────────────────────────────────
print("\nGenerating sample prediction grids...")
matcher = MedicineMatcher(brand_to_generic, accept_threshold=best_threshold)

correct_samples, incorrect_samples, uncertain_samples = [], [], []

for row in test_ds.items:
    img_path, gt_brand, gt_generic = row
    # find matching pred
    idx = test_ds.items.index(row)
    if idx >= len(test_preds):
        continue
    pred = test_preds[idx]
    conf = test_confs[idx]
    mr = matcher.match(pred, conf)
    sample = {
        "img_path": img_path,
        "pred": pred,
        "gt_brand": gt_brand,
        "matched_brand": mr.matched_brand,
        "status": mr.status,
        "confidence": mr.confidence,
    }
    if mr.status == "uncertain":
        uncertain_samples.append(sample)
    elif mr.matched_brand == gt_brand:
        correct_samples.append(sample)
    else:
        incorrect_samples.append(sample)


def save_sample_grid(samples, title, filename, n=12):
    if not samples:
        print(f"  No samples for: {title}")
        return
    samples = random.sample(samples, min(n, len(samples)))
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    if rows == 1:
        axes = [axes]
    for i, s in enumerate(samples):
        ax = axes[i // cols][i % cols]
        from PIL import Image
        with Image.open(s["img_path"]) as img:
            ax.imshow(np.array(img), cmap="gray", aspect="auto")
        status_color = {"correct": "green", "incorrect": "red", "uncertain": "orange"}.get(
            "correct" if s["status"] == "accepted" and s["matched_brand"] == s["gt_brand"]
            else "uncertain" if s["status"] == "uncertain"
            else "incorrect",
            "gray"
        )
        ax.set_title(
            f"GT: {s['gt_brand']}\nPred: {s['matched_brand']}\nConf: {s['confidence']:.2f}",
            fontsize=6, color=status_color
        )
        ax.axis("off")
    # Hide empty axes
    for j in range(len(samples), rows * cols):
        axes[j // cols][j % cols].axis("off")
    plt.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


save_sample_grid(correct_samples,   "Correct Predictions",   "samples_correct.png")
save_sample_grid(incorrect_samples, "Incorrect Predictions", "samples_incorrect.png")
save_sample_grid(uncertain_samples, "Uncertain Predictions", "samples_uncertain.png")

# Print summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Test CER          : {test_cer:.4f}")
print(f"Test brand acc    : {test_brand_acc:.4f}")
print(f"Test generic acc  : {test_generic_acc:.4f}")
print(f"Threshold (val)   : {best_threshold:.2f}")
print(f"Acceptance rate   : {ablation['trocr_with_matching']['acceptance_rate']:.4f}")
print(f"Acc (accepted)    : {ablation['trocr_with_matching']['brand_acc_among_accepted']:.4f}")
print(f"Acc (uncertain)   : {ablation['trocr_with_matching']['brand_acc_among_uncertain']:.4f}")
print(f"\nCorrect samples : {len(correct_samples)}")
print(f"Incorrect samples: {len(incorrect_samples)}")
print(f"Uncertain samples: {len(uncertain_samples)}")
print("=" * 60)
