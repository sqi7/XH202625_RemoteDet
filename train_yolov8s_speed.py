from ultralytics import YOLO
from pathlib import Path


def main():
    project_root = Path("/root/epfs/XH202625_RemoteDet")

    model_path = project_root / "weights" / "yolov8s.pt"
    data_yaml = project_root / "configs" / "xh202625_split.yaml"

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        imgsz=1024,
        epochs=100,
        batch=64,
        device=0,
        workers=8,
        cache=False,
        project=str(project_root / "runs" / "yolo"),
        name="E02_yolov8s_speed_b64_nocache",
        pretrained=True,
        optimizer="auto",
        patience=25,
        amp=True,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()

