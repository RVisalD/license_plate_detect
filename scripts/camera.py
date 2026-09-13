from pathlib import Path
from ultralytics import YOLO
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.3,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Live License Plate Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()