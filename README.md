# Medical Prescription Recognition Pipeline

This repository contains a full machine-learning pipeline for recognising handwritten medicine names from prescription word-crops.

## Features

- **Preprocessing & Augmentation**: Advanced OpenCV filtering (CLAHE, median blur) to restore faded ink, plus TorchVision augmentations (perspective, rotation) tailored for handwriting.
- **TrOCR Fine-tuning**: Fine-tunes Microsoft's TrOCR (Transformer-based Optical Character Recognition) to read cursive handwriting.
- **Mask R-CNN Scaffold**: Architecture ready for region detection (word-crop passthrough mode enabled by default since the current dataset only provides cropped words).
- **Fuzzy Dictionary Matching**: Uses Levenshtein distance and RapidFuzz to correct OCR errors by matching against a known database of 78 medicine brands and mapping them to their generic equivalents.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. Place your data in `data/raw/` (or run `python setup_data.py` if you have the original `archive__6_.zip` to partition it into train/val/test splits).
2. Ensure the directory structure looks like this:
```
data/
└── raw/
    ├── train/
    │   ├── images/
    │   └── labels.csv
    ├── val/
    │   ├── images/
    │   └── labels.csv
    └── test/
        ├── images/
        └── labels.csv
```

## Usage

### 1. Exploratory Data Analysis (EDA)
Check the dataset balance and verify there is no data leakage between splits using Perceptual Hashing.
```bash
python eda.py
```
*Results are saved in the `results/` folder.*

### 2. Train the OCR Model
Fine-tune the TrOCR model on the training set. This script supports early stopping and logs validation CER (Character Error Rate).
```bash
python train_trocr.py
```
*The best model is saved to `checkpoints/trocr_best/`.*

### 3. Evaluate the Pipeline
Evaluates the model on the `test` split. First tunes the optimal confidence threshold on the `val` split, then applies dictionary matching and calculates ablation metrics.
```bash
python evaluate.py
```
*Generates `results/metrics.json` and a full ablation table.*

### 4. Run Inference on New Images
Run the end-to-end pipeline on a single image or a folder of images.
```bash
python pipeline.py --image path/to/image.png
# OR
python pipeline.py --image path/to/folder/
```
*Outputs are appended to `results/predictions.csv` and `results/predictions.jsonl`.*

## Project Structure
- `common.py` - Core preprocessing, data augmentation, and dataset loading logic.
- `train_trocr.py` - TrOCR training loop with evaluation callbacks.
- `train_maskrcnn.py` - Mask R-CNN scaffold (passthrough mode).
- `matcher.py` - Fuzzy dictionary matching and confidence computation.
- `evaluate.py` - Full test-set evaluation and metrics generation.
- `pipeline.py` - Inference CLI for end-to-end structured prediction.
- `REPORT.md` - Technical report template.
