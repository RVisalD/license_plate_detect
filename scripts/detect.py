from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from crop_plate import crop_detected_boxes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "images" / "test" / "Cars145.png"


def load_model(model_path: str | Path = MODEL_PATH) -> YOLO:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def detect_image(source: str | Path, conf: float = 0.5) -> list[dict]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    model = load_model()
    results = model.predict(source=str(source_path), conf=conf, save=True, verbose=False)

    detections: list[dict] = []
    for result in results:
        if len(result.boxes) == 0:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "confidence": confidence,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

    return detections


def main(source: str | Path = DEFAULT_SOURCE) -> int:
    source_path = Path(source)
    print(f"Detecting license plate in: {source_path}")

    detections = detect_image(source_path)

    if not detections:
        print("No license plate detected.")
        return 0

    print("Plate detected!")
    for index, detection in enumerate(detections, start=1):
        print(f"Detection {index}:")
        print(f"  Confidence: {detection['confidence']:.2f}")
        print(
            "  Box: "
            f"{detection['x1']:.0f}, {detection['y1']:.0f}, "
            f"{detection['x2']:.0f}, {detection['y2']:.0f}"
        )

    choice = input("Do you want to crop this plate? (y/n): ").strip().lower()
    if choice in {"y", "yes"}:
        cropped_paths = crop_detected_boxes(source_path, detections, output_dir=PROJECT_ROOT / "cropped")
        print("Cropped plate(s) saved here:")
        for path in cropped_paths:
            print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
