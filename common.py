"""
common.py — Shared utilities: random seeds, preprocessing, augmentation, dataset class.
Used by every other script in the pipeline.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

for _d in [RESULTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── GPU / Device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TROCR_MODEL_NAME = (
    "microsoft/trocr-base-handwritten"
    if torch.cuda.is_available()
    else "microsoft/trocr-small-handwritten"
)


def get_device_info() -> str:
    if DEVICE.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU only — using trocr-small-handwritten"


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> Image.Image:
    """
    Load and preprocess an image:
      BGR → Grayscale → Median blur (k=3) → CLAHE → 3-channel RGB PIL Image.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    rgb3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb3ch)


# ─── Augmentation (training only) ─────────────────────────────────────────────
def get_train_augmentation():
    """
    Returns a callable that applies data augmentation to a PIL Image.
    All operations are mild to avoid destroying handwriting legibility.
    """

    def augment(img: Image.Image) -> Image.Image:
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # Small rotation ±5°
        angle = random.uniform(-5, 5)
        img = TF.rotate(img, angle, fill=255)

        # Random perspective
        if random.random() < 0.3:
            # RandomPerspective returns a transform which we apply to img
            persp = T.RandomPerspective(distortion_scale=0.2, p=1.0, fill=255)
            img = persp(img)

        # Brightness / contrast
        img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))

        # Gaussian blur
        if random.random() < 0.3:
            from PIL import ImageFilter
            radius = random.uniform(0.3, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Slight scale + padding
        w, h = img.size
        pad = random.randint(0, 4)
        img = TF.pad(img, pad, fill=255)
        img = img.resize((w, h), Image.LANCZOS)

        return img

    return augment


TRAIN_AUGMENT = get_train_augmentation()


# ─── Dataset ──────────────────────────────────────────────────────────────────
import csv


class PrescriptionDataset(torch.utils.data.Dataset):
    """
    Word-crop dataset. Each item is a (preprocessed PIL Image, label_str) tuple.
    Pass augment=True for training split only.
    """

    def __init__(self, split_dir: str, augment: bool = False):
        self.augment = augment
        self.images_dir = None
        self.items = []  # [(img_path, medicine_name, generic_name), ...]

        # Auto-detect image folder name
        for candidate in ["training_words", "validation_words", "testing_words", "images"]:
            d = os.path.join(split_dir, candidate)
            if os.path.isdir(d):
                self.images_dir = d
                break
        if self.images_dir is None:
            raise ValueError(f"No *_words directory found in {split_dir}")

        # Read CSV
        csv_files = [f for f in os.listdir(split_dir) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV found in {split_dir}")
        csv_path = os.path.join(split_dir, csv_files[0])

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_file = row["IMAGE"]
                img_path = os.path.join(self.images_dir, img_file)
                if os.path.exists(img_path):
                    self.items.append(
                        (img_path, row["MEDICINE_NAME"], row["GENERIC_NAME"])
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, med_name, gen_name = self.items[idx]
        img = preprocess_image(img_path)
        if self.augment:
            img = TRAIN_AUGMENT(img)
        return img, med_name, gen_name


def load_split(split: str, augment: bool = False) -> PrescriptionDataset:
    """Load train/val/test split from ./data/raw/{split}/"""
    split_dir = os.path.join(DATA_ROOT, split)
    return PrescriptionDataset(split_dir, augment=augment)


# ─── Brand → Generic map from training CSV ────────────────────────────────────
def load_brand_generic_map() -> dict:
    """Returns {brand_name: generic_name} from training CSV."""
    train_dir = os.path.join(DATA_ROOT, "train")
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    csv_path = os.path.join(train_dir, csv_files[0])
    bg_map = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bg_map[row["MEDICINE_NAME"]] = row["GENERIC_NAME"]
    return bg_map
