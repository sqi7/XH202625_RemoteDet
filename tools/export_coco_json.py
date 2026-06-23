import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
CLASS_NAMES = [
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
]


def find_images(image_dir):
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def parse_pred_line(line):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    score = float(parts[1])
    x1, y1, x2, y2 = map(float, parts[2:6])
    w = x2 - x1
    h = y2 - y1
    return cls, score, [x1, y1, w, h]


def load_image_size(img_path):
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--score_thr", type=float, default=0.0)
    parser.add_argument("--category_id_offset", type=int, default=0)
    parser.add_argument("--include_empty_images", action="store_true")
    parser.add_argument("--full_coco", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    pred_dir = Path(args.pred_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir)
    detections = []
    raw_predictions = 0
    filtered_by_score = 0
    invalid_boxes = 0
    class_counter = Counter()
    image_has_det = set()

    for img_path in images:
        pred_path = pred_dir / f"{img_path.stem}.txt"
        if not pred_path.exists():
            continue

        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            raw_predictions += 1
            cls, score, bbox = parse_pred_line(line)

            if score < args.score_thr:
                filtered_by_score += 1
                continue
            if bbox[2] <= 0 or bbox[3] <= 0:
                invalid_boxes += 1
                continue

            category_id = cls + args.category_id_offset
            det = {
                "image_id": img_path.stem,
                "category_id": category_id,
                "bbox": [round(v, 2) for v in bbox],
                "score": round(score, 6),
            }
            detections.append(det)
            class_counter[category_id] += 1
            image_has_det.add(img_path.stem)

    if args.full_coco:
        coco_images = []
        for img_path in images:
            if not args.include_empty_images and img_path.stem not in image_has_det:
                continue
            width, height = load_image_size(img_path)
            coco_images.append(
                {
                    "id": img_path.stem,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                }
            )

        annotations = []
        for ann_id, det in enumerate(detections, 1):
            bbox = det["bbox"]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": det["image_id"],
                    "category_id": det["category_id"],
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                    "score": det["score"],
                }
            )

        categories = [
            {"id": idx + args.category_id_offset, "name": name}
            for idx, name in enumerate(CLASS_NAMES)
        ]
        output = {"images": coco_images, "annotations": annotations, "categories": categories}
    else:
        output = detections

    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 80)
    print("COCO JSON export done")
    print(f"image_dir: {image_dir}")
    print(f"pred_dir: {pred_dir}")
    print(f"out_json: {out_json}")
    print(f"full_coco: {args.full_coco}")
    print(f"include_empty_images: {args.include_empty_images}")
    print(f"images: {len(images)}")
    print(f"images_with_detections: {len(image_has_det)}")
    print(f"raw_predictions: {raw_predictions}")
    print(f"filtered_by_score: {filtered_by_score}")
    print(f"invalid_boxes: {invalid_boxes}")
    print(f"exported_predictions: {len(detections)}")
    print("predictions_by_category_id:")
    for cls in sorted(class_counter):
        print(f"  {cls}: {class_counter[cls]}")


if __name__ == "__main__":
    main()
