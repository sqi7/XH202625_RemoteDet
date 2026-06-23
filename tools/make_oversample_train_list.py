from pathlib import Path
from collections import Counter

ROOT = Path("/root/epfs/XH202625_RemoteDet")
IMG_DIR = ROOT / "official_data/prepared/xh202625_yolo/images/train"
LAB_DIR = ROOT / "official_data/prepared/xh202625_yolo/labels/train"
OUT = ROOT / "official_data/prepared/xh202625_yolo/train_oversample.txt"

# 少样本/困难类别权重
# 含有这些类别的图像会被重复加入训练列表
CLASS_REPEAT = {
    0: 8,    # HM 极少
    1: 6,    # LQS 极少
    21: 3,   # KC-10 少样本
    24: 4,   # FSC 发射车，重点救
    2: 2,    # QHS 效果一般，轻微加强
}

def find_img(stem):
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

lines = []
repeat_counter = Counter()
image_counter = 0

for lab in sorted(LAB_DIR.glob("*.txt")):
    img = find_img(lab.stem)
    if img is None:
        continue

    image_counter += 1

    classes = set()
    for line in lab.read_text().splitlines():
        if line.strip():
            cls = int(float(line.split()[0]))
            classes.add(cls)

    repeat = 1
    for cls in classes:
        repeat = max(repeat, CLASS_REPEAT.get(cls, 1))

    for _ in range(repeat):
        lines.append(str(img))

    for cls in classes:
        repeat_counter[cls] += repeat

OUT.write_text("\n".join(lines) + "\n")

print(f"original train images: {image_counter}")
print(f"oversampled train lines: {len(lines)}")
print(f"saved to: {OUT}")
print("weighted image appearances by class:")
for cls in sorted(repeat_counter):
    print(cls, repeat_counter[cls])
