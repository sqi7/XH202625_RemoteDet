import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms_numpy(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = compute_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def classwise_nms(boxes, scores, classes, iou_thr=0.5):
    keep_all = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        keep = nms_numpy(boxes[idx], scores[idx], iou_thr)
        keep_all.extend(idx[keep].tolist())
    return keep_all


def valid_tile(tile, black_thr=10, max_black_ratio=0.75):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    black_ratio = (gray < black_thr).mean()
    return black_ratio <= max_black_ratio


def iter_tiles(img, tile_size=1024, overlap=0.2):
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    stride = max(1, stride)

    ys = list(range(0, max(h - tile_size + 1, 1), stride))
    xs = list(range(0, max(w - tile_size + 1, 1), stride))

    if len(ys) == 0 or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))
    if len(xs) == 0 or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))

    for y in ys:
        for x in xs:
            tile = img[y:y + tile_size, x:x + tile_size]
            yield x, y, tile


def draw_boxes(img, boxes, scores, classes, names, save_path):
    vis = img.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(int(cls), str(int(cls)))} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(save_path), vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="runs/sliced_infer")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--black_ratio", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names

    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.source)

    all_boxes = []
    all_scores = []
    all_classes = []

    tile_count = 0
    used_tile_count = 0

    for x0, y0, tile in iter_tiles(img, tile_size=args.tile, overlap=args.overlap):
        tile_count += 1

        if not valid_tile(tile, max_black_ratio=args.black_ratio):
            continue

        used_tile_count += 1

        results = model.predict(
            source=tile,
            imgsz=args.tile,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0

        all_boxes.append(xyxy)
        all_scores.append(confs)
        all_classes.append(clss)

    if all_boxes:
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)

        keep = classwise_nms(boxes, scores, classes, iou_thr=args.iou)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)

    stem = Path(args.source).stem
    txt_path = out_dir / f"{stem}.txt"
    vis_path = out_dir / f"{stem}_vis.jpg"

    with open(txt_path, "w", encoding="utf-8") as f:
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            f.write(f"{int(cls)} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    draw_boxes(img, boxes, scores, classes, names, vis_path)

    print("image:", args.source)
    print("total tiles:", tile_count)
    print("used tiles:", used_tile_count)
    print("detections:", len(boxes))
    print("txt saved:", txt_path)
    print("vis saved:", vis_path)


if __name__ == "__main__":
    main()
