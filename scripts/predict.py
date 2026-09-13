from pathlib import Path
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = PROJECT_ROOT / "best.pt"
SOURCE_IMAGE = PROJECT_ROOT / "images" / "train" / "Cars433.png"

# Check files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SOURCE_IMAGE.exists():
    raise FileNotFoundError(f"Image not found: {SOURCE_IMAGE}")

# Load trained model
model = YOLO(MODEL_PATH)

# Predict
results = model.predict(
    source=SOURCE_IMAGE,
    conf=0.5,
    save=True
)

# Print detections
for result in results:

    if len(result.boxes) == 0:
        print("No license plate detected.")
        continue

    for box in result.boxes:
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print("Plate detected!")
        print(f"Confidence: {confidence:.2f}")
        print(f"Box: {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")