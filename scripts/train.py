from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train on your license-plate dataset
model.train(
    data="../data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    save=True,
    save_period=5,

    project="runs",
    name="license_plate_detector"
)