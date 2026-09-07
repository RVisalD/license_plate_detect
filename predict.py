from ultralytics import YOLO

source_image = "images/train/Cars433.png"

model = YOLO(
    "runs/detect/runs/license_plate_finetuned/weights/best.pt"
)

results = model.predict(
    source=source_image,
    conf=0.5,
    save=True
)

for result in results:
    for box in result.boxes:

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print("Plate detected!")
        print("Confidence:", confidence)
        print("Box:", x1, y1, x2, y2)