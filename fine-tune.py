from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/license_plate_detector/weights/best.pt"
)

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    save=True,
    save_period=5,
    project="runs",
    name="license_plate_finetuned"
)