"""
pipeline.py — End-to-end inference CLI.

Usage:
  python pipeline.py --image <path_to_image_or_folder>
  python pipeline.py --image data/raw/test/images/0.png
  python pipeline.py --image data/raw/test/images/

Output appended to:
  results/predictions.csv
  results/predictions.jsonl

Each record:
  image, raw_text, matched_brand, generic_name,
  ocr_confidence, match_score, confidence, status,
  patient_name (empty), symptoms (empty)
"""

import os, sys, argparse, csv, json, math
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    set_seed, SEED, DEVICE, RESULTS_DIR, CHECKPOINTS_DIR,
    preprocess_image, load_brand_generic_map
)
from train_maskrcnn import MaskRCNNWrapper
from matcher import MedicineMatcher, build_medicine_dict

set_seed(SEED)

# ─── Load best threshold from results (written by evaluate.py tuning) ─────────
THRESHOLD_FILE = os.path.join(RESULTS_DIR, "best_threshold.json")
DEFAULT_THRESHOLD = 0.5

def load_threshold() -> float:
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return json.load(f).get("threshold", DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD


# ─── Lazy model loading ───────────────────────────────────────────────────────
_processor = None
_model = None

def load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model

    best_ckpt = os.path.join(CHECKPOINTS_DIR, "trocr_best")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"No trained checkpoint found at {best_ckpt}. "
            "Run train_trocr.py first."
        )

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    print(f"Loading TrOCR from {best_ckpt}...")
    _processor = TrOCRProcessor.from_pretrained(best_ckpt)
    _model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(DEVICE)
    _model.eval()
    return _processor, _model


# ─── OCR inference with confidence ───────────────────────────────────────────
def run_trocr(pil_image: Image.Image) -> tuple:
    """Returns (predicted_text: str, ocr_confidence: float)."""
    processor, model = load_model()

    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=32,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode text
    pred_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Compute sequence confidence from beam scores
    # scores: tuple of (vocab_size,) tensors, one per generated token
    try:
        if outputs.scores:
            log_probs = []
            for step_scores in outputs.scores:
                # step_scores shape: (1, vocab_size) — take the chosen token
                chosen_id = outputs.sequences[0, len(outputs.sequences[0]) - len(outputs.scores) + outputs.scores.index(step_scores)]
                probs = torch.softmax(step_scores[0], dim=-1)
                lp = torch.log(probs[outputs.sequences[0, -len(outputs.scores) + outputs.scores.index(step_scores)]] + 1e-9)
                log_probs.append(lp.item())
            length = max(len(log_probs), 1)
            ocr_conf = math.exp(sum(log_probs) / length)
        else:
            ocr_conf = 1.0
    except Exception:
        ocr_conf = 1.0

    return pred_text, float(ocr_conf)


# ─── Simple beam-score confidence (robust version) ────────────────────────────
def run_trocr_robust(pil_image: Image.Image) -> tuple:
    """Returns (predicted_text, ocr_confidence) with robust confidence estimate."""
    processor, model = load_model()
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=32,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
        )

    pred_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Robust confidence: mean of max softmax prob per step
    try:
        if outputs.scores:
            max_probs = []
            for step_scores in outputs.scores:
                probs = torch.softmax(step_scores[0], dim=-1)
                max_probs.append(probs.max().item())
            ocr_conf = float(sum(max_probs) / max(len(max_probs), 1))
        else:
            ocr_conf = 1.0
    except Exception:
        ocr_conf = 1.0

    return pred_text, ocr_conf


# ─── Predict single image ─────────────────────────────────────────────────────
def predict_image(image_path: str, matcher: MedicineMatcher, maskrcnn: MaskRCNNWrapper) -> dict:
    """Full pipeline for one image."""
    # 1. Preprocess
    pil_img = preprocess_image(image_path)

    # 2. Mask R-CNN (passthrough mode for word crops)
    detection = maskrcnn.detect(pil_img)
    # In passthrough mode, crop = full image (box covers entire image)

    # 3. TrOCR
    raw_text, ocr_conf = run_trocr_robust(pil_img)

    # 4. Dictionary matching
    mr = matcher.match(raw_text, ocr_conf)

    return {
        "image": os.path.abspath(image_path),
        "raw_text": raw_text,
        "matched_brand": mr.matched_brand,
        "generic_name": mr.generic_name,
        "ocr_confidence": round(ocr_conf, 4),
        "match_score": round(mr.match_score, 4),
        "confidence": round(mr.confidence, 4),
        "status": mr.status,
        "patient_name": "",    # Not available in this dataset
        "symptoms": "",        # Not available in this dataset
        "maskrcnn_mode": detection.get("mode", "unknown"),
    }


# ─── Output writers ───────────────────────────────────────────────────────────
CSV_PATH  = os.path.join(RESULTS_DIR, "predictions.csv")
JSONL_PATH = os.path.join(RESULTS_DIR, "predictions.jsonl")

CSV_FIELDS = [
    "image", "raw_text", "matched_brand", "generic_name",
    "ocr_confidence", "match_score", "confidence", "status",
    "patient_name", "symptoms", "maskrcnn_mode"
]


def write_record(record: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})

    # JSONL
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Main CLI ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prescription recognition pipeline")
    parser.add_argument("--image", required=True, help="Path to image file or folder of images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override accept threshold (default: load from results/best_threshold.json or 0.5)")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else load_threshold()
    print(f"Accept threshold: {threshold:.2f}")

    brand_generic_map = build_medicine_dict()
    matcher = MedicineMatcher(brand_generic_map, accept_threshold=threshold)
    maskrcnn = MaskRCNNWrapper(mode="passthrough")

    image_path_arg = args.image
    if os.path.isdir(image_path_arg):
        img_files = sorted(
            [os.path.join(image_path_arg, f)
             for f in os.listdir(image_path_arg)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
    else:
        img_files = [image_path_arg]

    print(f"\nProcessing {len(img_files)} image(s)...\n")
    for img_path in img_files:
        try:
            record = predict_image(img_path, matcher, maskrcnn)
            write_record(record)
            print(f"[{record['status'].upper():9s}] {os.path.basename(img_path)}")
            print(f"  raw_text     : {record['raw_text']!r}")
            print(f"  matched_brand: {record['matched_brand']}")
            print(f"  generic_name : {record['generic_name']}")
            print(f"  confidence   : {record['confidence']:.4f}")
            print()
        except Exception as e:
            print(f"  ERROR on {img_path}: {e}")

    print(f"Records written to:\n  {CSV_PATH}\n  {JSONL_PATH}")


if __name__ == "__main__":
    main()
