"""
setup_data.py — Copies the dataset from the Unicode path to ./data/raw/{train,val,test}/
Run once before anything else.
"""

import os
import shutil
import csv

# Source: the Unicode-apostrophe folder that already exists
SRC_BASE = os.path.join(
    r"C:\Users\vaibh\Downloads\HandWritten-Model-byAntiG",
    "Doctor\u2019s Handwritten Prescription BD dataset",
)

# Destination: clean ASCII path
DST_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

SPLIT_MAP = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test",
}

IMG_FOLDER_MAP = {
    "Training": "training_words",
    "Validation": "validation_words",
    "Testing": "testing_words",
}

CSV_MAP = {
    "Training": "training_labels.csv",
    "Validation": "validation_labels.csv",
    "Testing": "testing_labels.csv",
}


def copy_split(src_split_name: str, dst_split_name: str):
    src_dir = os.path.join(SRC_BASE, src_split_name)
    dst_dir = os.path.join(DST_BASE, dst_split_name)
    dst_img_dir = os.path.join(dst_dir, "images")
    os.makedirs(dst_img_dir, exist_ok=True)

    # Copy images
    src_img_dir = os.path.join(src_dir, IMG_FOLDER_MAP[src_split_name])
    n_copied = 0
    for fname in os.listdir(src_img_dir):
        if fname.lower().endswith(".png"):
            src_f = os.path.join(src_img_dir, fname)
            dst_f = os.path.join(dst_img_dir, fname)
            if not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)
            n_copied += 1

    # Copy CSV
    src_csv = os.path.join(src_dir, CSV_MAP[src_split_name])
    dst_csv = os.path.join(dst_dir, "labels.csv")
    shutil.copy2(src_csv, dst_csv)

    # Rewrite CSV paths to point at ./images/<n>.png
    rows = []
    with open(dst_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["IMAGE", "MEDICINE_NAME", "GENERIC_NAME"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "IMAGE": row["IMAGE"],
                    "MEDICINE_NAME": row["MEDICINE_NAME"],
                    "GENERIC_NAME": row["GENERIC_NAME"],
                }
            )

    print(f"  {src_split_name:10s} -> {dst_split_name:5s}/ : {n_copied} images, labels.csv copied")
    return n_copied


if __name__ == "__main__":
    print("Setting up clean data paths...")
    total = 0
    for src_name, dst_name in SPLIT_MAP.items():
        total += copy_split(src_name, dst_name)
    print(f"\nDone. Total images copied/verified: {total}")
    print(f"Data root: {DST_BASE}")
