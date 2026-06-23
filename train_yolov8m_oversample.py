from ultralytics import YOLO
from pathlib import Path

def main():
    root = Path("/root/epfs/XH202625_RemoteDet")

    model = YOLO(str(root / "weights/yolov8m.pt"))

    model.train(
        data=str(root / "configs/xh202625_oversample.yaml"),
        imgsz=1024,
        epochs=100,
        batch=16,
        device=0,
        workers=8,
        cache=False,
        project=str(root / "runs/yolo"),
        name="E04_yolov8m_oversample",
        pretrained=True,
        optimizer="auto",
        patience=30,
        amp=True,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
