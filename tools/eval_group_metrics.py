import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
DEFAULT_GROUPS = ["ship:0-3", "aircraft:4-23", "vehicle:24"]
DEFAULT_GROUP_IOU = ["ship:0.5", "aircraft:0.5", "vehicle:0.35"]


def parse_class_spec(spec):
    classes = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            classes.extend(range(int(start), int(end) + 1))
        else:
            classes.append(int(item))
    return classes


def parse_groups(items):
    groups = []
    class_to_group = {}

    for item in items:
        if ":" not in item:
            raise ValueError(f"invalid group spec: {item}, expected name:0,1,2 or name:0-3")
        name, spec = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty group name in spec: {item}")

        classes = parse_class_spec(spec)
        if not classes:
            raise ValueError(f"empty class list in spec: {item}")

        groups.append(name)
        for cls in classes:
            if cls in class_to_group:
                raise ValueError(f"class {cls} appears in multiple groups")
            class_to_group[cls] = name

    return groups, class_to_group


def parse_group_iou(items, default_iou):
    group_iou = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"invalid group_iou spec: {item}, expected name:iou")
        name, iou = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty group name in group_iou spec: {item}")
        group_iou[name] = float(iou)
    return group_iou


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


def find_images(image_dir):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(image_dir.glob(ext))
    return sorted(image_paths)


def load_gt(label_path, img_w, img_h, class_to_group):
    gts = []
    if not label_path.exists():
        return gts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, box = yolo_to_xyxy(line, img_w, img_h)
        group = class_to_group.get(cls, f"unknown_{cls}")
        gts.append({"cls": cls, "group": group, "box": box, "matched": False})
    return gts


def load_preds(pred_path, class_to_group):
    preds = []
    if not pred_path.exists():
        return preds

    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, score, box = pred_to_xyxy(line)
        group = class_to_group.get(cls, f"unknown_{cls}")
        preds.append({"cls": cls, "group": group, "score": score, "box": box})

    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def load_image_size(img_path):
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--group_iou",
        nargs="*",
        default=DEFAULT_GROUP_IOU,
        help="per-group IoU specs, e.g. ship:0.5 aircraft:0.5 vehicle:0.35",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=DEFAULT_GROUPS,
        help="group specs, e.g. ship:0-3 aircraft:4-23 vehicle:24",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    pred_dir = Path(args.pred_dir)

    group_names, class_to_group = parse_groups(args.groups)
    group_iou = parse_group_iou(args.group_iou, args.iou)
    stats = defaultdict(lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0})
    for name in group_names:
        stats[name]

    image_paths = find_images(image_dir)
    skipped_images = 0

    for img_path in image_paths:
        w, h = load_image_size(img_path)
        if w is None or h is None:
            skipped_images += 1
            continue

        stem = img_path.stem

        gts = load_gt(label_dir / f"{stem}.txt", w, h, class_to_group)
        preds = load_preds(pred_dir / f"{stem}.txt", class_to_group)

        for gt in gts:
            stats[gt["group"]]["gt"] += 1
        for pred in preds:
            stats[pred["group"]]["pred"] += 1

        for pred in preds:
            best_iou = 0.0
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

            match_iou = group_iou.get(pred["group"], args.iou)
            if best_iou >= match_iou and best_idx >= 0:
                gts[best_idx]["matched"] = True
                stats[pred["group"]]["tp"] += 1
            else:
                stats[pred["group"]]["fp"] += 1

        for gt in gts:
            if not gt["matched"]:
                stats[gt["group"]]["fn"] += 1

    total = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    for d in stats.values():
        for key in total:
            total[key] += d[key]

    overall_recall = total["tp"] / (total["tp"] + total["fn"] + 1e-6)
    overall_fdr = total["fp"] / (total["tp"] + total["fp"] + 1e-6)

    print("=" * 80)
    print("Group metrics")
    print(f"image_dir: {image_dir}")
    print(f"label_dir: {label_dir}")
    print(f"pred_dir: {pred_dir}")
    print(f"default_iou: {args.iou}")
    print(f"group_iou: {group_iou}")
    print(f"images: {len(image_paths)}")
    print(f"skipped_images: {skipped_images}")
    print()
    print("Overall")
    print(f"GT: {total['gt']}")
    print(f"Pred: {total['pred']}")
    print(f"TP: {total['tp']}")
    print(f"FP: {total['fp']}")
    print(f"FN: {total['fn']}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"FDR = FP / (FP + TP): {overall_fdr:.4f}")

    print("\nPer-group")
    print("group\tIoU\tGT\tPred\tTP\tFP\tFN\tRecall\tFDR")

    ordered_groups = group_names + sorted(k for k in stats.keys() if k not in group_names)
    for group in ordered_groups:
        d = stats[group]
        iou_thr = group_iou.get(group, args.iou)
        recall = d["tp"] / (d["tp"] + d["fn"] + 1e-6)
        fdr = d["fp"] / (d["tp"] + d["fp"] + 1e-6)
        print(
            f"{group}\t{iou_thr:.2f}\t{d['gt']}\t{d['pred']}\t{d['tp']}\t"
            f"{d['fp']}\t{d['fn']}\t{recall:.4f}\t{fdr:.4f}"
        )


if __name__ == "__main__":
    main()
