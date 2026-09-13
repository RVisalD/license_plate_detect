from pathlib import Path
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
DATA_PATH = PROJECT_ROOT / "data.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"

# Check required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"data.yaml not found: {DATA_PATH}")

# Load pretrained YOLO model
model = YOLO(MODEL_PATH)

# Train on license plate dataset
model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    save_period=5,
    project=RUNS_PATH,
    name="license_plate_detector"
)