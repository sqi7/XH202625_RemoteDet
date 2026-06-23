from ultralytics import YOLO
from pathlib import Path


def main():
    project_root = Path("/root/epfs/XH202625_RemoteDet")

    model_path = project_root / "weights" / "yolov8m.pt"
    data_yaml = project_root / "configs" / "xh202625_split.yaml"

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        imgsz=1024,
        epochs=100,
        batch=16,
        device=0,
        workers=8,
        project=str(project_root / "runs" / "yolo"),
        name="E01_yolov8m_baseline",
        pretrained=True,
        optimizer="auto",
        patience=30,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
