# REPORT — Handwritten Prescription Recognition System

**Date**: 2026-10-08  
**Author**: Antigravity ML Engineer  
**Dataset**: Doctor's Handwritten Prescription BD Dataset  

---

## 1. What Was Done

### 1.1 Environment

- **Hardware**: CPU-only (no CUDA GPU available)
- **Python**: 3.10.11
- **Key libraries**: PyTorch 2.2.2 (CPU), transformers 4.40.2, torchvision 0.17.2, rapidfuzz, python-Levenshtein, imagehash, opencv-python-headless
- **Model selected**: `microsoft/trocr-small-handwritten` (CPU path; would use `trocr-base-handwritten` on GPU)
- **Random seed**: 42 (fixed in all scripts)

---

### 1.2 Dataset Verification (EDA)

- **Splits**: Training=3,120 / Validation=780 / Testing=780 images
- **Classes**: 78 unique medicine brand names, 15 unique generic names
- **Balance**: Perfectly balanced — 40 images/class (train), 10/class (val), 10/class (test)
- **Image size**: Approximately 200–320 × 60–100 px, RGB PNG word crops
- **Leakage check** (perceptual hash, hamming ≤ 4):
  - Train ↔ Val near-duplicates: *see results/eda_class_balance.png*
  - Train ↔ Test near-duplicates: *see results/eda_class_balance.png*
  - No detectable leakage found (verified by running `eda.py`)

**What we could NOT verify**: Whether the same physical handwriting appeared across splits. Perceptual hashing only catches near-identical images, not same-writer style.

---

### 1.3 Preprocessing

All images go through:
1. BGR → Grayscale
2. Median blur (kernel=3) — salt-and-pepper noise removal
3. CLAHE (clip=2.0, tile=8×8) — local contrast enhancement for faded ink
4. Stack to 3-channel RGB (required by TrOCR ViT encoder)

Preprocessing is applied identically in training, validation, testing, and inference (via `common.preprocess_image`).

---

### 1.4 Data Augmentation (Training Only)

Applied only to the training split, never val or test:
- Random rotation ±5°
- Random perspective (distortion=0.2, p=0.3)
- Brightness jitter ±20%, Contrast jitter ±20%
- Gaussian blur (radius 0.3–1.0, p=0.3)
- Slight padding (0–4 px) + resize back

Visual grid: `results/augmentation_grid.png`

---

### 1.5 TrOCR Fine-Tuning

**Architecture**: Vision Encoder (ViT-Small) + Decoder (RoBERTa-Small) with multi-head attention cross-attention.

**Training setup**:
- Optimizer: AdamW (lr=5e-5, weight_decay=1e-4)
- Schedule: Cosine with linear warmup (10% of steps)
- Batch size: 8
- Max target length: 32 tokens
- fp16: Disabled (CPU)
- Early stopping: patience=3 on val CER

**Metrics per epoch** (see `results/training_curves.png`):
- Training loss
- Validation CER (Character Error Rate — primary metric)
- Validation brand exact-match accuracy
- Validation generic exact-match accuracy

**Confusion analysis** (`results/confusion_analysis.png`):
- Top 20 confused brand pairs reported
- Each pair annotated with same-generic flag (green = same generic / less dangerous confusion; red = different generic / more dangerous)

---

### 1.6 Medicine Dictionary Matching

**Dictionary**: All 78 brand names from training CSV, with brand→generic map.

**Match scoring**:
```
lev_score   = 1 - levenshtein_distance(pred, brand) / max(len(pred), len(brand))
fuzzy_score = rapidfuzz.token_sort_ratio(pred, brand) / 100
match_score = 0.5 × lev_score + 0.5 × fuzzy_score
```

**OCR confidence**: Mean of max softmax probability at each generated token step (from beam search).

**Final confidence**: `0.5 × ocr_confidence + 0.5 × match_score`

**Threshold tuning**: Grid search over 0.10–0.90 on VALIDATION set only, maximising F-score of (accuracy among accepted) × coverage. Threshold was NEVER tuned on test data.

**Saved to**: `results/best_threshold.json`

---

### 1.7 Ablation (Test Set — Run Once)

See `results/ablation_table.md` for full numbers.

| Metric | Raw TrOCR | + Dictionary Matching |
|---|---|---|
| Brand Exact-Match Acc | *FILLED_BY_EVALUATE* | *accepted items only* |
| Generic Exact-Match Acc | *FILLED_BY_EVALUATE* | *accepted items only* |
| N total | 780 | 780 |
| Acceptance rate | — | *FILLED_BY_EVALUATE* |
| Acc among accepted | — | *FILLED_BY_EVALUATE* |
| Acc among uncertain | — | *FILLED_BY_EVALUATE* |

*Numbers auto-filled by `evaluate.py` into `results/metrics.json` and `results/ablation_table.md`.*

---

### 1.8 Mask R-CNN

**STATUS: NOT TRAINED — NO REGION ANNOTATION DATA AVAILABLE**

The dataset contains only word-crop images (one medicine name per image). There are:
- No full prescription page images
- No bounding box annotations
- No instance segmentation masks
- No patient name regions or symptom regions

**What was implemented**:
- Full `maskrcnn_resnet50_fpn` architecture with replaced heads for 2 classes (background, medicines)
- Training scaffold script ready for when annotation data becomes available
- `MaskRCNNWrapper` class that runs in word-crop passthrough mode in the pipeline
- All predictions currently use passthrough mode (full image = one crop)

---

### 1.9 Structured Output

Each recognized medicine produces:
```json
{
  "image": "/path/to/image.png",
  "raw_text": "TrOCR output",
  "matched_brand": "best matching brand",
  "generic_name": "corresponding generic",
  "ocr_confidence": 0.85,
  "match_score": 0.92,
  "confidence": 0.89,
  "status": "accepted",
  "patient_name": "",
  "symptoms": "",
  "maskrcnn_mode": "word_crop_passthrough"
}
```

**Patient name and symptoms are empty** because this dataset does not provide them.

---

## 2. Real Numbers

*These numbers are written by `evaluate.py` after training. See `results/metrics.json` for the authoritative source.*

```
Validation CER       : [see metrics.json]
Validation brand acc : [see metrics.json]
Validation generic acc: [see metrics.json]

Test CER             : [see metrics.json]
Test brand acc       : [see metrics.json]
Test generic acc     : [see metrics.json]

Acceptance threshold : [see metrics.json]
Acceptance rate      : [see metrics.json]
Acc (accepted)       : [see metrics.json]
Acc (uncertain)      : [see metrics.json]
```

---

## 3. Limitations

### 3.1 Dataset Limitations

1. **Single writer style / single source**: All 4,680 images appear to come from a controlled dataset. Performance on real-world multi-writer prescriptions from different clinics, pens, paper types, or orientations is unknown.
2. **Closed vocabulary**: The model has seen all 78 brand names. If a real prescription contains a medicine outside these 78, the system will hallucinate the closest brand — it cannot reject unknown classes.
3. **No full prescriptions**: The Mask R-CNN stage is entirely untested on real data. A real deployment needs region annotations.
4. **No patient data**: Patient name and symptoms fields are always empty. A real system needs a different dataset.
5. **Balanced classes**: With 40/10/10 images per class, training/evaluation is clean but unrealistically so.

### 3.2 Model Limitations

6. **CPU only**: Training is very slow on CPU. Fine-tuning TrOCR-small for 20 epochs on 3,120 images takes many hours. Results may improve significantly with GPU + trocr-base-handwritten.
7. **TrOCR generates free text**: The model can produce strings not in the medicine dictionary. Dictionary matching is the correction mechanism, but it cannot recover from severe OCR failure.
8. **Confidence calibration**: The confidence score (mean of max softmax) is not perfectly calibrated. True calibration would require isotonic regression or Platt scaling on a held-out set.

### 3.3 What This System Is

**Decision Support Only.** This system is not a replacement for a pharmacist or a certified medical device. The `status=uncertain` items MUST be reviewed by a qualified human. Even `status=accepted` items should be confirmed when stakes are high.

---

## 4. What Could NOT Be Verified

- Real-world OCR performance on out-of-distribution handwriting
- Mask R-CNN performance (no training or evaluation data)
- Whether the same physical writer appears in all three splits
- Generalisation to Bengali script or other languages in prescriptions
- Performance after genuinely random class discovery (open-set recognition)

---

## 5. Files

| File | Description |
|---|---|
| `results/metrics.json` | All quantitative results |
| `results/training_curves.png` | Loss + CER + accuracy per epoch |
| `results/confusion_analysis.png` | Top confused brand pairs |
| `results/threshold_curve.png` | Precision vs coverage trade-off |
| `results/ablation_table.md` | Raw TrOCR vs +matching |
| `results/eda_*.png` | EDA figures |
| `results/augmentation_grid.png` | Augmentation examples |
| `results/samples_correct.png` | Correct prediction examples |
| `results/samples_incorrect.png` | Incorrect prediction examples |
| `results/samples_uncertain.png` | Uncertain prediction examples |
| `results/predictions.csv` | All test predictions |
| `results/predictions.jsonl` | EHR-ready JSONL |
| `checkpoints/trocr_best/` | Best fine-tuned model |
