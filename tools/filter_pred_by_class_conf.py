import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--default", type=float, default=0.25)
    parser.add_argument("--class_thr", nargs="*", default=[])
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    class_thr = {}
    for item in args.class_thr:
        cls, thr = item.split(":")
        class_thr[int(cls)] = float(thr)

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

if __name__ == "__main__":
    main()
