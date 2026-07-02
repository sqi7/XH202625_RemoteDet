from ultralytics import YOLO

def main():
    model = YOLO("runs/yolo/E05_dior_pretrain_yolov8m/weights/best.pt")

    model.train(
        data="configs/xh202625_split.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        workers=16,
        device=0,
        project="runs/yolo",
        name="E06_dior_finetune_xh202625_yolov8m",
        pretrained=True,
        patience=30,
        seed=0,
        deterministic=True,
        cache=False,
        amp=True,
    )

if __name__ == "__main__":
    main()
