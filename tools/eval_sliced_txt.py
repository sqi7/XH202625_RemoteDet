import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])

    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return cls, [x1, y1, x2, y2]


def pred_to_xyxy(line):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    score = float(parts[1])
    x1, y1, x2, y2 = map(float, parts[2:6])
    return cls, score, [x1, y1, x2, y2]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def load_gt(label_path, img_w, img_h):
    gts = []
    if not label_path.exists():
        return gts

    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, box = yolo_to_xyxy(line, img_w, img_h)
        gts.append({"cls": cls, "box": box, "matched": False})
    return gts


def load_preds(pred_path):
    preds = []
    if not pred_path.exists():
        return preds

    for line in pred_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, score, box = pred_to_xyxy(line)
        preds.append({"cls": cls, "score": score, "box": box})
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    pred_dir = Path(args.pred_dir)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        image_paths.extend(image_dir.glob(ext))
    image_paths = sorted(image_paths)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_cls = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0})

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        stem = img_path.stem

        gt_path = label_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"

        gts = load_gt(gt_path, w, h)
        preds = load_preds(pred_path)

        for gt in gts:
            per_cls[gt["cls"]]["gt"] += 1
        for pred in preds:
            per_cls[pred["cls"]]["pred"] += 1

        tp = 0
        fp = 0

        for pred in preds:
            best_iou = 0
            best_idx = -1

            for i, gt in enumerate(gts):
                if gt["matched"]:
                    continue
                if gt["cls"] != pred["cls"]:
                    continue

                cur_iou = iou_xyxy(pred["box"], gt["box"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = i

            if best_iou >= args.iou and best_idx >= 0:
                gts[best_idx]["matched"] = True
                tp += 1
                per_cls[pred["cls"]]["tp"] += 1
            else:
                fp += 1
                per_cls[pred["cls"]]["fp"] += 1

        fn = sum(1 for gt in gts if not gt["matched"])
        for gt in gts:
            if not gt["matched"]:
                per_cls[gt["cls"]]["fn"] += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    far = 1 - precision

    print("=" * 80)
    print("Overall")
    print(f"Images: {len(image_paths)}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"False Alarm Rate = 1 - Precision: {far:.4f}")

    print("\nPer-class")
    print("cls\tGT\tPred\tTP\tFP\tFN\tP\tR")
    for cls in sorted(per_cls.keys()):
        d = per_cls[cls]
        p = d["tp"] / (d["tp"] + d["fp"] + 1e-6)
        r = d["tp"] / (d["tp"] + d["fn"] + 1e-6)
        print(f"{cls}\t{d['gt']}\t{d['pred']}\t{d['tp']}\t{d['fp']}\t{d['fn']}\t{p:.4f}\t{r:.4f}")


if __name__ == "__main__":
    main()
