import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(source_dir):
    source_dir = Path(source_dir)
    return sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter

    return inter / np.maximum(union, 1e-9)


def nms_numpy(dets, iou_thr=0.5):
    """
    dets: Nx6, columns = cls, score, x1, y1, x2, y2
    class-wise NMS
    """
    if len(dets) == 0:
        return dets

    dets = np.array(dets, dtype=np.float32)
    keep_all = []

    classes = np.unique(dets[:, 0]).astype(int)

    for cls in classes:
        cls_dets = dets[dets[:, 0] == cls]
        order = np.argsort(-cls_dets[:, 1])
        cls_dets = cls_dets[order]

        keep = []
        while len(cls_dets) > 0:
            cur = cls_dets[0]
            keep.append(cur)

            if len(cls_dets) == 1:
                break

            rest = cls_dets[1:]
            ious = iou_xyxy(cur[2:6], rest[:, 2:6])
            cls_dets = rest[ious < iou_thr]

        keep_all.extend(keep)

    keep_all = np.array(keep_all, dtype=np.float32)
    order = np.argsort(-keep_all[:, 1])
    return keep_all[order]


def iter_tiles(img, tile_size, overlap):
    h, w = img.shape[:2]
    step = int(tile_size * (1.0 - overlap))
    step = max(1, step)

    xs = list(range(0, max(w - tile_size, 0) + 1, step))
    ys = list(range(0, max(h - tile_size, 0) + 1, step))

    if not xs or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))
    if not ys or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))

    seen = set()
    for y in ys:
        for x in xs:
            if (x, y) in seen:
                continue
            seen.add((x, y))

            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]

            yield x, y, tile


def is_black_or_empty(tile, black_thr=3.0):
    if tile.size == 0:
        return True
    return float(tile.mean()) < black_thr


def draw_vis(img, dets, out_path):
    vis = img.copy()
    for det in dets:
        cls, score, x1, y1, x2, y2 = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{int(cls)} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_det", type=int, default=3000)
    parser.add_argument("--skip_black", action="store_true")
    parser.add_argument("--black_thr", type=float, default=3.0)
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.source_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    print(f"model: {args.model}")
    print(f"source_dir: {args.source_dir}")
    print(f"out: {out_dir}")
    print(f"images: {len(images)}")
    print(f"tile: {args.tile}")
    print(f"overlap: {args.overlap}")
    print(f"conf: {args.conf}")
    print(f"iou_nms: {args.iou}")
    print(f"device: {args.device}")

    model = YOLO(args.model)

    per_image = []
    total_tiles = 0
    used_tiles = 0
    total_detections = 0

    t_all0 = time.perf_counter()

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        img_dets = []
        image_tiles = 0
        image_used_tiles = 0

        t0 = time.perf_counter()

        for ox, oy, tile in iter_tiles(img, args.tile, args.overlap):
            image_tiles += 1
            total_tiles += 1

            if args.skip_black and is_black_or_empty(tile, args.black_thr):
                continue

            image_used_tiles += 1
            used_tiles += 1

            pred_imgsz = args.imgsz if args.imgsz is not None else args.tile

            results = model.predict(
                tile,
                imgsz=pred_imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )

            if not results:
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for box, score, cls in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()

                x1 += ox
                x2 += ox
                y1 += oy
                y2 += oy

                x1 = max(0, min(float(x1), w))
                x2 = max(0, min(float(x2), w))
                y1 = max(0, min(float(y1), h))
                y2 = max(0, min(float(y2), h))

                if x2 <= x1 or y2 <= y1:
                    continue

                img_dets.append([int(cls), float(score), x1, y1, x2, y2])

        if img_dets:
            img_dets = nms_numpy(img_dets, args.iou)
        else:
            img_dets = np.zeros((0, 6), dtype=np.float32)

        total_detections += len(img_dets)

        out_txt = out_dir / f"{img_path.stem}.txt"
        lines = []
        for det in img_dets:
            cls, score, x1, y1, x2, y2 = det
            lines.append(f"{int(cls)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))

        if args.save_vis:
            draw_vis(img, img_dets, out_dir / "vis" / f"{img_path.stem}.jpg")

        elapsed = time.perf_counter() - t0
        per_image.append(
            {
                "image": img_path.name,
                "width": w,
                "height": h,
                "time_sec": elapsed,
                "tiles": image_tiles,
                "used_tiles": image_used_tiles,
                "detections": int(len(img_dets)),
            }
        )

        print(
            f"[{idx}/{len(images)}] {img_path.name} "
            f"{w}x{h} time={elapsed:.3f}s "
            f"tiles={image_tiles} used={image_used_tiles} det={len(img_dets)}"
        )

    total_time = time.perf_counter() - t_all0
    avg_time = sum(x["time_sec"] for x in per_image) / len(per_image) if per_image else 0.0
    max_time = max((x["time_sec"] for x in per_image), default=0.0)

    summary = {
        "model": args.model,
        "source_dir": str(args.source_dir),
        "out": str(out_dir),
        "images": len(per_image),
        "total_time_sec": total_time,
        "avg_time_per_image_sec": avg_time,
        "max_time_per_image_sec": max_time,
        "total_tiles": total_tiles,
        "used_tiles": used_tiles,
        "total_detections": total_detections,
        "tile": args.tile,
        "overlap": args.overlap,
        "conf": args.conf,
        "iou": args.iou,
        "per_image": per_image,
    }

    (out_dir / "speed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print()
    print("=" * 80)
    print(f"images: {len(per_image)}")
    print(f"total_time: {total_time:.3f}s")
    print(f"avg_time_per_image: {avg_time:.3f}s")
    print(f"max_time_per_image: {max_time:.3f}s")
    print(f"total_tiles: {total_tiles}")
    print(f"used_tiles: {used_tiles}")
    print(f"total_detections: {total_detections}")
    print(f"speed_summary: {out_dir / 'speed_summary.json'}")


if __name__ == "__main__":
    main()
