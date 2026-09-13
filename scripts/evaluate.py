from pathlib import Path
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = PROJECT_ROOT / "best.pt"
DATA_PATH = PROJECT_ROOT / "data.yaml"

# Check files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"data.yaml not found: {DATA_PATH}")

# Load model
model = YOLO(MODEL_PATH)

# Evaluate on test dataset
metrics = model.val(
    data=DATA_PATH,
    split="test"
)

# Print results
print("\n--- Test Results ---")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)