import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("RUN:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Final test inference pipeline: sliced inference -> class-threshold filtering -> COCO JSON export"
    )

    parser.add_argument("--model", required=True, help="Path to final best.pt")
    parser.add_argument("--test_image_dir", required=True, help="Directory of official test images")
    parser.add_argument("--out_root", default="runs/final_test", help="Output root directory")
    parser.add_argument("--submit_json", default="results/FINAL_test_submission.json")

    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--default_thr", type=float, default=0.25)
    parser.add_argument(
        "--class_thr",
        nargs="*",
        default=["0:0.10", "1:0.10", "24:0.12"],
        help="Class-specific thresholds, e.g. 0:0.10 1:0.10 24:0.12",
    )

    parser.add_argument(
        "--category_id_offset",
        type=int,
        default=0,
        help="Use 0 for class ids 0-24, use 1 for class ids 1-25",
    )

    parser.add_argument(
        "--skip_infer",
        action="store_true",
        help="Skip sliced inference and only run filtering/export using existing raw predictions",
    )

    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip COCO JSON export",
    )

    args = parser.parse_args()

    out_root = Path(args.out_root)
    raw_dir = out_root / "raw_conf005"
    submit_dir = out_root / "submit_filtered"
    submit_json = Path(args.submit_json)

    out_root.mkdir(parents=True, exist_ok=True)
    submit_json.parent.mkdir(parents=True, exist_ok=True)

    print("\nFinal inference config")
    print(f"model: {args.model}")
    print(f"test_image_dir: {args.test_image_dir}")
    print(f"out_root: {out_root}")
    print(f"raw_dir: {raw_dir}")
    print(f"submit_dir: {submit_dir}")
    print(f"submit_json: {submit_json}")
    print(f"default_thr: {args.default_thr}")
    print(f"class_thr: {args.class_thr}")
    print(f"category_id_offset: {args.category_id_offset}")

    if not args.skip_infer:
        infer_cmd = [
            "python",
            "tools/infer_sliced_yolo_batch.py",
            "--model",
            args.model,
            "--source_dir",
            args.test_image_dir,
            "--out",
            str(raw_dir),
            "--tile",
            str(args.tile),
            "--overlap",
            str(args.overlap),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--device",
            str(args.device),
        ]

        if args.limit and args.limit > 0:
            infer_cmd += ["--limit", str(args.limit)]

        run_cmd(infer_cmd)
    else:
        print(f"\nSkip inference. Use existing raw predictions: {raw_dir}")

    filter_cmd = [
        "python",
        "tools/filter_pred_by_class_conf.py",
        "--src",
        str(raw_dir),
        "--dst",
        str(submit_dir),
        "--default",
        str(args.default_thr),
        "--class_thr",
        *args.class_thr,
    ]
    run_cmd(filter_cmd)

    if not args.skip_export:
        export_cmd = [
            "python",
            "tools/export_coco_json.py",
            "--image_dir",
            args.test_image_dir,
            "--pred_dir",
            str(submit_dir),
            "--out_json",
            str(submit_json),
            "--category_id_offset",
            str(args.category_id_offset),
        ]
        run_cmd(export_cmd)
    else:
        print("\nSkip COCO JSON export.")

    print("\nDONE")
    print(f"raw predictions: {raw_dir}")
    print(f"filtered predictions: {submit_dir}")
    if not args.skip_export:
        print(f"submission json: {submit_json}")
    print(f"speed summary: {raw_dir / 'speed_summary.json'}")


if __name__ == "__main__":
    main()


# conda activate remote_det
# cd ~/epfs/XH202625_RemoteDet

# python tools/final_infer_test.py \
#   --model runs/yolo/E04_yolov8m_oversample/weights/best.pt \
#   --test_image_dir official_data/test/images \
#   --out_root runs/final_test/E04_balance \
#   --submit_json results/E04_balance_submission.json \
#   --default_thr 0.25 \
#   --class_thr 0:0.10 1:0.10 24:0.12 \
#   --category_id_offset 0 \
#   --device 0