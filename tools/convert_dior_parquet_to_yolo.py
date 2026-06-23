import os
import json
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm


ROOT = Path("public_data/DIOR")
RAW_DATA = ROOT / "raw" / "data"
OUT = ROOT / "yolo"

# DIOR 20类，HuggingFace版本一般是0-19编号
# 类名只用于显示，不影响训练；后续如README中有更准顺序，可再修正
CLASS_NAMES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]


def ensure_dirs():
    for split in ["train", "val"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_parquet_files(split_name):
    if split_name == "train":
        return sorted(RAW_DATA.glob("train-*.parquet"))
    if split_name == "val":
        return sorted(RAW_DATA.glob("validation-*.parquet"))
    raise ValueError(split_name)


def save_one(row, split):
    image_id = str(row["image_id"])
    width = int(row["width"])
    height = int(row["height"])

    img_obj = row["image"]
    img_bytes = img_obj["bytes"]

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_path = OUT / "images" / split / f"{image_id}.jpg"
    img.save(img_path, quality=95)

    objects = row["objects"]
    bboxes = objects["bbox"]
    categories = objects["category"]

    label_lines = []
    for bbox, cat in zip(bboxes, categories):
        # 常见COCO格式：[x, y, w, h]
        x, y, w, h = map(float, bbox)

        # 防止异常框
        if w <= 0 or h <= 0:
            continue

        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        bw = w / width
        bh = h / height

        # clip到[0,1]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        cat = int(cat)
        label_lines.append(f"{cat} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    label_path = OUT / "labels" / split / f"{image_id}.txt"
    label_path.write_text("\n".join(label_lines), encoding="utf-8")


def convert_split(split):
    files = get_parquet_files(split)
    total = 0
    print(f"[{split}] parquet files:", len(files))

    for fp in files:
        df = pd.read_parquet(fp)
        print(f"processing {fp.name}, rows={len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            save_one(row, split)
            total += 1

    print(f"[{split}] saved images/labels:", total)


def write_yaml():
    yaml_text = f"""path: {OUT.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"

    (OUT / "dior.yaml").write_text(yaml_text, encoding="utf-8")


def write_classes():
    (OUT / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")


def main():
    ensure_dirs()
    convert_split("train")
    convert_split("val")
    write_yaml()
    write_classes()
    print("Done.")
    print("YOLO dir:", OUT.resolve())
    print("YAML:", (OUT / "dior.yaml").resolve())


if __name__ == "__main__":
    main()
