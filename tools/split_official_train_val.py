import random
import shutil
from pathlib import Path

random.seed(2026)

SRC = Path("official_data/extracted/data")
DST = Path("official_data/prepared/xh202625_yolo")

IMG_TRAIN = SRC / "images" / "train"
LAB_TRAIN = SRC / "labels" / "train"

OUT_IMG_TRAIN = DST / "images" / "train"
OUT_IMG_VAL = DST / "images" / "val"
OUT_LAB_TRAIN = DST / "labels" / "train"
OUT_LAB_VAL = DST / "labels" / "val"

for p in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LAB_TRAIN, OUT_LAB_VAL]:
    p.mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = sorted([p for p in IMG_TRAIN.iterdir() if p.suffix.lower() in img_exts])

valid_pairs = []
missing_labels = []

for img in images:
    lab = LAB_TRAIN / f"{img.stem}.txt"
    if lab.exists():
        valid_pairs.append((img, lab))
    else:
        missing_labels.append(img.name)

random.shuffle(valid_pairs)

val_ratio = 0.2
n_val = int(len(valid_pairs) * val_ratio)

val_pairs = valid_pairs[:n_val]
train_pairs = valid_pairs[n_val:]

def copy_pairs(pairs, img_out, lab_out):
    for img, lab in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lab, lab_out / lab.name)

copy_pairs(train_pairs, OUT_IMG_TRAIN, OUT_LAB_TRAIN)
copy_pairs(val_pairs, OUT_IMG_VAL, OUT_LAB_VAL)

print("Total valid pairs:", len(valid_pairs))
print("Train:", len(train_pairs))
print("Val:", len(val_pairs))
print("Missing labels:", len(missing_labels))
if missing_labels[:10]:
    print("Example missing labels:", missing_labels[:10])
