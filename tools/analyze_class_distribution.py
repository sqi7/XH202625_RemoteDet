from pathlib import Path
from collections import Counter

names = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "A1_SU-35", 5: "A2_C-130",
    6: "A3_C-17", 7: "A4_C-5", 8: "A5_F-16", 9: "A6_TU-160",
    10: "A7_E-3", 11: "A8_B-52", 12: "A9_P-3C", 13: "A10_B-1B",
    14: "A11_E-8", 15: "A12_TU-22", 16: "A13_F-15", 17: "A14_KC-135",
    18: "A15_F-22", 19: "A16_FA-18", 20: "A17_TU-95", 21: "A18_KC-10",
    22: "A19_SU-34", 23: "A20_SU-24", 24: "FSC"
}

def count_dir(label_dir):
    counter = Counter()
    img_counter = Counter()
    files = sorted(Path(label_dir).glob("*.txt"))
    for f in files:
        classes_in_img = set()
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(float(line.split()[0]))
            counter[cls] += 1
            classes_in_img.add(cls)
        for cls in classes_in_img:
            img_counter[cls] += 1
    return counter, img_counter, len(files)

for split in ["train", "val"]:
    label_dir = Path(f"official_data/prepared/xh202625_yolo/labels/{split}")
    obj_cnt, img_cnt, n_files = count_dir(label_dir)

    print("=" * 80)
    print(f"{split}: label files = {n_files}")
    print("cls\tname\tobjects\timages")
    for cls in range(25):
        print(f"{cls}\t{names[cls]}\t{obj_cnt[cls]}\t{img_cnt[cls]}")
