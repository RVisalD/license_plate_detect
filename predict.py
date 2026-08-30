from ultralytics import YOLO

source_image = "https://static01.nyt.com/images/2025/04/25/00xp-streetview-opening-pgzc-cover/00xp-streetview-opening-pgzc-cover-verticalTwoByThree735.jpg"

model = YOLO(
    "best.pt"
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