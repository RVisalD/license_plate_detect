from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/license_plate_detector/weights/best.pt"
)

metrics = model.val(
    data="../data.yaml",
    split="test"
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)