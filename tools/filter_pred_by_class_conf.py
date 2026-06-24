import argparse
from pathlib import Path
from collections import defaultdict

import cv2


def parse_class_thr(items):
    out = {}
    for item in items:
        cls, thr = item.split(":")
        out[int(cls)] = float(thr)
    return out


def parse_thresholds(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_groups(items):
    groups = {}
    cls_to_group = {}
    for item in items:
        name, spec = item.split(":")
        cls_ids = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                cls_ids.extend(list(range(int(a), int(b) + 1)))
            else:
                cls_ids.append(int(part))
        groups[name] = cls_ids
        for c in cls_ids:
            cls_to_group[c] = name
    return groups, cls_to_group


def parse_group_iou(items):
    out = {}
    for item in items:
        name, val = item.split(":")
        out[name] = float(val)
    return out


def xywhn_to_xyxy(line, img_w, img_h):
    parts = line.split()
    cls = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:5])
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return {"cls": cls, "box": [x1, y1, x2, y2], "matched": False}


def pred_line_to_obj(line):
    parts = line.split()
    if len(parts) < 6:
        return None
    return {
        "cls": int(float(parts[0])),
        "score": float(parts[1]),
        "box": list(map(float, parts[2:6])),
    }


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def find_image(image_dir, stem):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_gts_for_class(label_dir, image_dir, cls_id):
    gts_by_img = defaultdict(list)

    for label_path in sorted(label_dir.glob("*.txt")):
        img_path = find_image(image_dir, label_path.stem)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            gt = xywhn_to_xyxy(line, w, h)
            if gt["cls"] == cls_id:
                gts_by_img[label_path.stem].append(gt)

    return gts_by_img


def load_preds_for_class(pred_dir, cls_id, score_thr):
    preds = []

    for pred_path in sorted(pred_dir.glob("*.txt")):
        for line in pred_path.read_text().splitlines():
            if not line.strip():
                continue
            pred = pred_line_to_obj(line)
            if pred is None:
                continue
            if pred["cls"] == cls_id and pred["score"] >= score_thr:
                pred["image_id"] = pred_path.stem
                preds.append(pred)

    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def eval_one_class(pred_dir, label_dir, image_dir, cls_id, score_thr, iou_thr):
    gts_by_img = load_gts_for_class(label_dir, image_dir, cls_id)
    preds = load_preds_for_class(pred_dir, cls_id, score_thr)

    total_gt = sum(len(v) for v in gts_by_img.values())
    tp = 0
    fp = 0

    for pred in preds:
        img_id = pred["image_id"]
        candidates = gts_by_img.get(img_id, [])

        best_iou = 0.0
        best_idx = -1

        for i, gt in enumerate(candidates):
            if gt["matched"]:
                continue

            cur_iou = iou_xyxy(pred["box"], gt["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = i

        if best_idx >= 0 and best_iou >= iou_thr:
            candidates[best_idx]["matched"] = True
            tp += 1
        else:
            fp += 1

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "cls": cls_id,
        "thr": score_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fdr": fdr,
    }


def run_filter(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    class_thr = parse_class_thr(args.class_thr)

    txts = sorted(src.glob("*.txt"))
    total_in = 0
    total_out = 0

    for txt in txts:
        out_lines = []
        lines = txt.read_text().splitlines()

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()
            cls = int(float(parts[0]))
            score = float(parts[1])
            thr = class_thr.get(cls, args.default)

            total_in += 1
            if score >= thr:
                out_lines.append(line)
                total_out += 1

        (dst / txt.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"default_thr: {args.default}")
    print(f"class_thr: {class_thr}")
    print(f"total_in: {total_in}")
    print(f"total_out: {total_out}")


def run_search(args):
    pred_dir = Path(args.src)
    label_dir = Path(args.label_dir)
    image_dir = Path(args.image_dir)

    _, cls_to_group = parse_groups(args.groups)
    group_iou = parse_group_iou(args.group_iou)
    thresholds = parse_thresholds(args.thresholds)

    if args.search_classes:
        search_classes = [int(x) for x in args.search_classes]
    else:
        search_classes = sorted(cls_to_group.keys())

    print(f"pred_dir: {pred_dir}")
    print(f"label_dir: {label_dir}")
    print(f"image_dir: {image_dir}")
    print(f"thresholds: {thresholds}")
    print()

    suggestions = {}

    for cls_id in search_classes:
        group = cls_to_group.get(cls_id, "unknown")
        iou_thr = group_iou.get(group, args.iou)

        print("=" * 80)
        print(f"class {cls_id} | group={group} | iou_thr={iou_thr}")
        print("thr\tTP\tFP\tFN\tP\tR\tFDR")

        best = None
        all_results = []

        for thr in thresholds:
            res = eval_one_class(pred_dir, label_dir, image_dir, cls_id, thr, iou_thr)
            all_results.append(res)
            print(
                f"{thr:.2f}\t{res['tp']}\t{res['fp']}\t{res['fn']}\t"
                f"{res['precision']:.4f}\t{res['recall']:.4f}\t{res['fdr']:.4f}"
            )

            if res["fdr"] <= args.target_fdr:
                if best is None:
                    best = res
                else:
                    if res["recall"] > best["recall"]:
                        best = res
                    elif res["recall"] == best["recall"] and res["precision"] > best["precision"]:
                        best = res

        if best is None:
            best = max(all_results, key=lambda x: (x["recall"], x["precision"]))

        suggestions[cls_id] = best["thr"]
        print(f"best_suggest: class {cls_id}:{best['thr']:.2f}  "
              f"P={best['precision']:.4f} R={best['recall']:.4f} FDR={best['fdr']:.4f}")

    print()
    print("=" * 80)
    print("Suggested class_thr:")
    print(" ".join([f"{c}:{thr:.2f}" for c, thr in suggestions.items()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="raw prediction txt directory")
    parser.add_argument("--dst", default=None, help="filtered prediction txt directory")
    parser.add_argument("--default", type=float, default=0.25)
    parser.add_argument("--class_thr", nargs="*", default=[])

    parser.add_argument("--search", action="store_true", help="search threshold for selected classes")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--search_classes", nargs="*", default=[])
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--target_fdr", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["ship:0-3", "aircraft:4-23", "vehicle:24"],
    )
    parser.add_argument(
        "--group_iou",
        nargs="*",
        default=["ship:0.5", "aircraft:0.5", "vehicle:0.35"],
    )

    args = parser.parse_args()

    if args.search:
        if args.image_dir is None or args.label_dir is None:
            raise ValueError("--search requires --image_dir and --label_dir")
        run_search(args)
    else:
        if args.dst is None:
            raise ValueError("filter mode requires --dst")
        run_filter(args)


if __name__ == "__main__":
    main()
