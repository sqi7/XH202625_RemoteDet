import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iter_tiles(img, tile_size=1024, overlap=0.2):
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    stride = max(1, stride)

    xs = list(range(0, max(w - tile_size + 1, 1), stride))
    ys = list(range(0, max(h - tile_size + 1, 1), stride))

    if not xs or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))
    if not ys or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))

    for y in ys:
        for x in xs:
            yield x, y, img[y:y + tile_size, x:x + tile_size]


def black_ratio(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return float((gray < 10).mean())


def nms_classwise(boxes, scores, classes, iou_thr=0.5):
    keep = []

    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        b = boxes[idxs]
        s = scores[idxs]
        order = s.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            keep.append(idxs[i])

            if len(order) == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(b[i, 0], b[rest, 0])
            yy1 = np.maximum(b[i, 1], b[rest, 1])
            xx2 = np.minimum(b[i, 2], b[rest, 2])
            yy2 = np.minimum(b[i, 3], b[rest, 3])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_r = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-6)

            order = rest[iou <= iou_thr]

    return keep


def draw(img, boxes, scores, classes, names, out_path):
    vis = img.copy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        name = names.get(int(cls), str(int(cls)))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{name} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(out_path), vis)


def infer_one_image(model, img_path, out_dir, tile_size, overlap, conf, iou, device, max_black, save_vis=True):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    all_boxes = []
    all_scores = []
    all_classes = []

    total_tiles = 0
    used_tiles = 0

    for x0, y0, tile in iter_tiles(img, tile_size=tile_size, overlap=overlap):
        total_tiles += 1

        if black_ratio(tile) > max_black:
            continue

        used_tiles += 1

        result = model.predict(
            source=tile,
            imgsz=tile_size,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

    if all_boxes:
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)

        keep = nms_classwise(boxes, scores, classes, iou_thr=iou)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)

    stem = img_path.stem
    txt_path = out_dir / f"{stem}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            f.write(f"{int(cls)} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    if save_vis:
        vis_path = out_dir / f"{stem}_vis.jpg"
        draw(img, boxes, scores, classes, model.names, vis_path)

    return {
        "image": str(img_path),
        "total_tiles": total_tiles,
        "used_tiles": used_tiles,
        "detections": len(boxes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_black", type=float, default=0.75)
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        images.extend(source_dir.glob(ext))
    images = sorted(images)

    if args.limit > 0:
        images = images[:args.limit]

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    total_time = 0.0
    total_det = 0
    total_tiles = 0
    total_used_tiles = 0

    for i, img_path in enumerate(images, 1):
        start = time.time()

        info = infer_one_image(
            model=model,
            img_path=img_path,
            out_dir=out_dir,
            tile_size=args.tile,
            overlap=args.overlap,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_black=args.max_black,
            save_vis=args.save_vis,
        )

        elapsed = time.time() - start

        total_time += elapsed
        total_det += info["detections"]
        total_tiles += info["total_tiles"]
        total_used_tiles += info["used_tiles"]

        print(
            f"[{i}/{len(images)}] {img_path.name} "
            f"time={elapsed:.3f}s tiles={info['used_tiles']}/{info['total_tiles']} det={info['detections']}"
        )

    print("=" * 70)
    print(f"images: {len(images)}")
    print(f"total_time: {total_time:.3f}s")
    print(f"avg_time_per_image: {total_time / max(len(images), 1):.3f}s")
    print(f"total_tiles: {total_tiles}")
    print(f"used_tiles: {total_used_tiles}")
    print(f"total_detections: {total_det}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
