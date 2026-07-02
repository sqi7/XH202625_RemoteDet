from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="public_data/DIOR/yolo/dior.yaml",
        imgsz=640,
        epochs=50,
        batch=32,
        workers=16,
        device=0,
        project="runs/yolo",
        name="E05_dior_pretrain_yolov8m",
        pretrained=True,
        patience=15,
        seed=0,
        deterministic=True,
        cache=False,
        amp=True,
    )

if __name__ == "__main__":
    main()
