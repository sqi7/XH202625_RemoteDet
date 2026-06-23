import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_images(image_dir):
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    x1 = min(max(x1, 0.0), img_w)
    y1 = min(max(y1, 0.0), img_h)
    x2 = min(max(x2, 0.0), img_w)
    y2 = min(max(y2, 0.0), img_h)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return f"{cls} {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def load_labels(label_path, orig_w, orig_h, x_offset, y_offset, scale_x, scale_y, mosaic_size):
    lines = []
    class_counter = Counter()

    if not label_path.exists():
        return lines, class_counter

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, x1, y1, x2, y2 = yolo_to_xyxy(line, orig_w, orig_h)
        x1 = x1 * scale_x + x_offset
        x2 = x2 * scale_x + x_offset
        y1 = y1 * scale_y + y_offset
        y2 = y2 * scale_y + y_offset

        out_line = xyxy_to_yolo(cls, x1, y1, x2, y2, mosaic_size, mosaic_size)
        if out_line is None:
            continue

        lines.append(out_line)
        class_counter[cls] += 1

    return lines, class_counter


def choose_images(images, count, rng):
    if len(images) >= count:
        return rng.sample(images, count)
    return [rng.choice(images) for _ in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--out_image_dir", required=True)
    parser.add_argument("--out_label_dir", required=True)
    parser.add_argument("--mosaic_size", type=int, default=10000)
    parser.add_argument("--tile_size", type=int, default=1000)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split_prefix", default="mosaic")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_image_dir = Path(args.out_image_dir)
    out_label_dir = Path(args.out_label_dir)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if args.mosaic_size % args.tile_size != 0:
        raise ValueError("--mosaic_size must be divisible by --tile_size")

    grid = args.mosaic_size // args.tile_size
    tiles_per_mosaic = grid * grid
    images = find_images(image_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    rng = random.Random(args.seed)

    total_mosaics = 0
    total_tiles = 0
    total_bad_tiles = 0
    total_objects = 0
    class_counter = Counter()

    for mosaic_idx in range(args.num):
        selected = choose_images(images, tiles_per_mosaic, rng)
        mosaic = Image.new("RGB", (args.mosaic_size, args.mosaic_size), (0, 0, 0))
        label_lines = []

        for tile_idx, img_path in enumerate(selected):
            row = tile_idx // grid
            col = tile_idx % grid
            x0 = col * args.tile_size
            y0 = row * args.tile_size

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                total_bad_tiles += 1
                continue

            orig_w, orig_h = img.size
            resized = img.resize((args.tile_size, args.tile_size), Image.BILINEAR)
            mosaic.paste(resized, (x0, y0))

            scale_x = args.tile_size / orig_w
            scale_y = args.tile_size / orig_h
            labels, counts = load_labels(
                label_path=label_dir / f"{img_path.stem}.txt",
                orig_w=orig_w,
                orig_h=orig_h,
                x_offset=x0,
                y_offset=y0,
                scale_x=scale_x,
                scale_y=scale_y,
                mosaic_size=args.mosaic_size,
            )
            label_lines.extend(labels)
            class_counter.update(counts)

        out_stem = f"{args.split_prefix}_{mosaic_idx:04d}"
        out_img = out_image_dir / f"{out_stem}.jpg"
        out_lab = out_label_dir / f"{out_stem}.txt"
        mosaic.save(out_img, quality=95)
        out_lab.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        total_mosaics += 1
        total_tiles += len(selected)
        total_objects += len(label_lines)

        print(f"[{mosaic_idx + 1}/{args.num}] {out_img.name} tiles={len(selected)} objects={len(label_lines)}")

    print("=" * 80)
    print("Mosaic generation done")
    print(f"source_images: {len(images)}")
    print(f"mosaic_size: {args.mosaic_size}")
    print(f"tile_size: {args.tile_size}")
    print(f"grid: {grid}x{grid}")
    print(f"generated_mosaics: {total_mosaics}")
    print(f"used_tiles: {total_tiles}")
    print(f"bad_tiles: {total_bad_tiles}")
    print(f"total_objects: {total_objects}")
    print("objects_by_class:")
    for cls in sorted(class_counter):
        print(f"  {cls}: {class_counter[cls]}")
    print(f"out_image_dir: {out_image_dir}")
    print(f"out_label_dir: {out_label_dir}")


if __name__ == "__main__":
    main()
