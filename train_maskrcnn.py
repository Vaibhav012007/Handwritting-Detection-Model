"""
train_maskrcnn.py — Mask R-CNN scaffold for full-prescription region detection.

STATUS: No region annotation data available in this dataset.
        This script implements the model architecture and training loop but
        DOES NOT TRAIN on the word-crop dataset (which has no masks/boxes).

        Run this script to confirm the architecture is wired correctly.
        Training requires:
          - Full prescription page images (not available)
          - Instance segmentation masks for each region (patient_name, symptoms, medicines)

        In the pipeline, this module runs in "word-crop passthrough mode":
        it returns a fake full-image bounding box and passes the whole crop
        directly to TrOCR.

SYNTHETIC stretch: Not implemented in this version (marked for future work).
"""

import os, sys
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, CHECKPOINTS_DIR

# ─── Class definitions ────────────────────────────────────────────────────────
# 0 = background, 1 = medicines
# (patient_name=2, symptoms=3 would require annotation data not present here)
NUM_CLASSES = 2  # background + medicines
CLASS_NAMES = ["background", "medicines"]


def build_maskrcnn_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Build a Mask R-CNN model with replaced classification and mask heads.
    Uses pretrained ResNet-50-FPN backbone.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_word_crop_passthrough(image_pil):
    """
    Word-crop passthrough mode: since there are no full prescription pages,
    this function returns a pseudo-detection covering the entire image.
    """
    from PIL import Image
    import numpy as np

    if isinstance(image_pil, Image.Image):
        w, h = image_pil.size
    else:
        h, w = image_pil.shape[:2]

    return {
        "boxes": torch.tensor([[0, 0, w, h]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),   # 1 = medicines
        "scores": torch.tensor([1.0], dtype=torch.float32),
        "mode": "word_crop_passthrough",
    }


# ─── Training scaffold (requires annotation data not available) ───────────────
def train_maskrcnn(
    annotation_dir: str,
    num_epochs: int = 10,
    lr: float = 5e-4,
):
    """
    Training loop scaffold. Call this only when you have:
      annotation_dir/
        images/*.jpg        (full prescription pages)
        annotations.json    (COCO-format instance segmentation)

    This dataset does NOT provide this, so this function raises immediately.
    """
    raise NotImplementedError(
        "\n"
        "=" * 70 + "\n"
        "Mask R-CNN TRAINING SKIPPED\n"
        "=" * 70 + "\n"
        "Reason: The dataset contains only word-crop images with medicine\n"
        "names. There are NO full prescription page images, NO bounding\n"
        "box annotations, and NO instance segmentation masks.\n\n"
        "To train Mask R-CNN you would need:\n"
        "  1. Full prescription page images\n"
        "  2. COCO-format instance segmentation masks for each region\n"
        "     (patient_name, symptoms, medicines)\n\n"
        "The pipeline uses 'word-crop passthrough mode' instead.\n"
        "=" * 70
    )


# ─── Inference wrapper ────────────────────────────────────────────────────────
class MaskRCNNWrapper:
    """
    Wraps Mask R-CNN for use in pipeline.py.
    Currently always runs in word-crop passthrough mode.
    If a trained checkpoint is present, it can be loaded for future use.
    """

    def __init__(self, checkpoint_path: str = None, mode: str = "passthrough"):
        self.mode = mode
        self.model = None

        if mode == "model" and checkpoint_path and os.path.exists(checkpoint_path):
            self.model = build_maskrcnn_model().to(DEVICE)
            state = torch.load(checkpoint_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"Loaded Mask R-CNN from {checkpoint_path}")
        else:
            print(
                "[MaskRCNN] Running in WORD-CROP PASSTHROUGH MODE — "
                "no detection performed (no region annotations available)."
            )

    def detect(self, image_pil):
        if self.mode == "passthrough" or self.model is None:
            return get_word_crop_passthrough(image_pil)

        import torchvision.transforms.functional as TF
        img_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = self.model(img_tensor)[0]
        return preds


if __name__ == "__main__":
    print("Building Mask R-CNN model (architecture check)...")
    model = build_maskrcnn_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Classes: {CLASS_NAMES}")
    print()
    print("NOTE: Training is NOT possible with this dataset.")
    print("      This script confirms the architecture is wired correctly.")
    print("      The pipeline uses word-crop passthrough mode.")
    print()
    try:
        train_maskrcnn("")
    except NotImplementedError as e:
        print(e)
