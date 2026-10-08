"""
train_trocr.py — Fine-tune TrOCR on the prescription word-crop dataset.

Metrics reported each epoch: CER, brand-exact-match, generic-exact-match.
Best checkpoint saved by validation CER with early stopping (patience=3).
Produces:
  checkpoints/trocr_best/        — best model weights + processor
  results/training_curves.png    — loss + CER curves
  results/confusion_analysis.png — top confused brand pairs
  results/confusion_data.json    — raw confusion data
"""

import os, sys, json, csv, math, time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    set_seed, SEED, DEVICE, TROCR_MODEL_NAME,
    DATA_ROOT, RESULTS_DIR, CHECKPOINTS_DIR,
    load_split, load_brand_generic_map, get_device_info
)

set_seed(SEED)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
MAX_TARGET_LENGTH = 32
BATCH_SIZE = 8      # conservative for CPU RAM
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 3
GRADIENT_CLIP = 1.0
USE_FP16 = DEVICE.type == "cuda"

print(f"Device: {get_device_info()}")
print(f"Model : {TROCR_MODEL_NAME}")
print(f"fp16  : {USE_FP16}")

# ─── Load processor and model ─────────────────────────────────────────────────
print("\nLoading processor and model (may download on first run)...")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.max_length = MAX_TARGET_LENGTH
model.config.no_repeat_ngram_size = 0
model.config.length_penalty = 1.0
model.config.num_beams = 4

model.to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── Brand → generic map ──────────────────────────────────────────────────────
brand_to_generic = load_brand_generic_map()

# ─── Custom collate ───────────────────────────────────────────────────────────
def collate_fn(batch):
    images, med_names, gen_names = zip(*batch)
    # Processor handles resizing internally
    encoding = processor(images=list(images), return_tensors="pt", padding=True)
    # Tokenise labels
    label_enc = processor.tokenizer(
        list(med_names),
        return_tensors="pt",
        padding=True,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    labels = label_enc.input_ids
    # Replace padding token id with -100 so it is ignored in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return encoding.pixel_values, labels, list(med_names), list(gen_names)

# ─── Datasets & loaders ───────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = load_split("train", augment=True)
val_ds   = load_split("val",   augment=False)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=False
)

# ─── CER helper ───────────────────────────────────────────────────────────────
def char_error_rate(preds, gts):
    """Compute mean CER over a batch (using edit distance)."""
    import Levenshtein as lev
    errors, total = 0, 0
    for p, g in zip(preds, gts):
        errors += lev.distance(p, g)
        total += max(len(g), 1)
    return errors / total if total > 0 else 0.0

# ─── Optimizer & scheduler ────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

# ─── Evaluation function ──────────────────────────────────────────────────────
def evaluate(loader, desc="Val"):
    model.eval()
    total_cer = 0.0
    brand_correct = 0
    generic_correct = 0
    n = 0
    all_preds, all_gt_brands, all_gt_generics = [], [], []

    with torch.no_grad():
        for pixel_values, labels, med_names, gen_names in tqdm(loader, desc=desc, leave=False):
            pixel_values = pixel_values.to(DEVICE)
            generated_ids = model.generate(
                pixel_values,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            preds = [p.strip() for p in preds]

            total_cer += char_error_rate(preds, med_names) * len(preds)
            brand_correct += sum(p == g for p, g in zip(preds, med_names))
            generic_correct += sum(
                brand_to_generic.get(p, "?") == brand_to_generic.get(g, "!")
                for p, g in zip(preds, med_names)
            )
            n += len(preds)
            all_preds.extend(preds)
            all_gt_brands.extend(med_names)
            all_gt_generics.extend(gen_names)

    return {
        "cer": total_cer / n if n > 0 else 1.0,
        "brand_acc": brand_correct / n if n > 0 else 0.0,
        "generic_acc": generic_correct / n if n > 0 else 0.0,
        "n": n,
        "preds": all_preds,
        "gt_brands": all_gt_brands,
        "gt_generics": all_gt_generics,
    }

# ─── Training loop ────────────────────────────────────────────────────────────
history = {"epoch": [], "train_loss": [], "val_cer": [], "val_brand_acc": [], "val_generic_acc": []}
best_val_cer = float("inf")
patience_counter = 0
best_ckpt_dir = os.path.join(CHECKPOINTS_DIR, "trocr_best")

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for pixel_values, labels, _, _ in tqdm(train_loader, desc=f"Ep {epoch}/{NUM_EPOCHS}", leave=False):
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        if USE_FP16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    elapsed = time.time() - t0

    # Validate
    val_metrics = evaluate(val_loader, desc=f"Val Ep{epoch}")
    val_cer = val_metrics["cer"]
    val_brand_acc = val_metrics["brand_acc"]
    val_generic_acc = val_metrics["generic_acc"]

    history["epoch"].append(epoch)
    history["train_loss"].append(avg_loss)
    history["val_cer"].append(val_cer)
    history["val_brand_acc"].append(val_brand_acc)
    history["val_generic_acc"].append(val_generic_acc)

    print(
        f"Ep {epoch:2d}/{NUM_EPOCHS}  "
        f"loss={avg_loss:.4f}  "
        f"val_CER={val_cer:.4f}  "
        f"brand_acc={val_brand_acc:.4f}  "
        f"generic_acc={val_generic_acc:.4f}  "
        f"[{elapsed:.0f}s]"
    )

    # Save best checkpoint
    if val_cer < best_val_cer:
        best_val_cer = val_cer
        patience_counter = 0
        os.makedirs(best_ckpt_dir, exist_ok=True)
        model.save_pretrained(best_ckpt_dir)
        processor.save_pretrained(best_ckpt_dir)
        print(f"  [OK] Best checkpoint saved (val_CER={val_cer:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience {patience_counter}/{EARLY_STOP_PATIENCE}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# ─── Training curves ──────────────────────────────────────────────────────────
epochs = history["epoch"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, history["train_loss"], marker="o", color="#e85d5d", label="train loss")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(epochs, history["val_cer"], marker="o", color="#4a90d9", label="val CER")
axes[1].set_title("Validation CER (lower=better)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

axes[2].plot(epochs, history["val_brand_acc"], marker="o", label="brand acc")
axes[2].plot(epochs, history["val_generic_acc"], marker="s", linestyle="--", label="generic acc")
axes[2].set_title("Validation Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].legend()

plt.tight_layout()
curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=120)
plt.close()
print(f"\nSaved training curves: {curve_path}")

# Save training history JSON
hist_path = os.path.join(RESULTS_DIR, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)

# ─── Confusion analysis (using best model on val) ─────────────────────────────
print("\nRunning confusion analysis on validation set with best checkpoint...")
best_model = VisionEncoderDecoderModel.from_pretrained(best_ckpt_dir).to(DEVICE)
best_model.eval()

val_metrics_best = evaluate(
    DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0),
    desc="Confusion analysis"
)

preds = val_metrics_best["preds"]
gt_brands = val_metrics_best["gt_brands"]

# Count confused pairs
from collections import Counter
confused = Counter()
for p, g in zip(preds, gt_brands):
    if p != g:
        confused[(g, p)] += 1

top_confused = confused.most_common(20)

# Annotate with same-generic flag
print("\nTop 20 confused brand pairs (ground_truth -> predicted):")
print(f"{'GT brand':<20} {'Pred brand':<20} {'Count':>6}  {'Same generic?'}")
for (gt_b, pred_b), cnt in top_confused:
    gt_gen = brand_to_generic.get(gt_b, "?")
    pred_gen = brand_to_generic.get(pred_b, "?")
    same = "[OK] YES" if gt_gen == pred_gen else "[X] NO "
    print(f"  {gt_b:<18} -> {pred_b:<18} {cnt:>6}  [{same}]  ({gt_gen} / {pred_gen})")

# Save confusion data
confusion_data = {
    "top_confused": [
        {
            "gt_brand": gt_b,
            "pred_brand": pred_b,
            "count": cnt,
            "gt_generic": brand_to_generic.get(gt_b, "?"),
            "pred_generic": brand_to_generic.get(pred_b, "?"),
            "same_generic": brand_to_generic.get(gt_b, "?") == brand_to_generic.get(pred_b, "?"),
        }
        for (gt_b, pred_b), cnt in top_confused
    ]
}
cdata_path = os.path.join(RESULTS_DIR, "confusion_data.json")
with open(cdata_path, "w") as f:
    json.dump(confusion_data, f, indent=2)

# Plot confusion bar chart
if top_confused:
    labels_plot = [f"{g}→{p}" for (g, p), _ in top_confused]
    counts_plot = [c for _, c in top_confused]
    colors = [
        "#27ae60" if brand_to_generic.get(g, "?") == brand_to_generic.get(p, "!")
        else "#e74c3c"
        for (g, p), _ in top_confused
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(labels_plot[::-1], counts_plot[::-1], color=colors[::-1])
    ax.set_xlabel("# Errors")
    ax.set_title("Top Confused Brand Pairs (green = same generic)")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Same generic (less dangerous)"),
        Patch(facecolor="#e74c3c", label="Different generic (more dangerous)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    conf_plot_path = os.path.join(RESULTS_DIR, "confusion_analysis.png")
    plt.savefig(conf_plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {conf_plot_path}")

print(f"\nBest val CER : {best_val_cer:.4f}")
print(f"Best val brand acc : {val_metrics_best['brand_acc']:.4f}")
print(f"Best val generic acc: {val_metrics_best['generic_acc']:.4f}")
print("\nTraining complete.")
