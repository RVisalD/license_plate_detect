from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def crop_box(image_path: str | Path, box, output_dir: str | Path | None = None, index: int = 0) -> Path:
    """Crop a single detection box from an image and save it."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop coordinates for {image_path}: {box}")

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError(f"Crop is empty for {image_path}: {box}")

    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "cropped"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_plate_{index}.png"
    ok = cv2.imwrite(str(output_path), cropped)
    if not ok:
        raise OSError(f"Failed to save crop to {output_path}")

    return output_path


def crop_detected_boxes(image_path: str | Path, detections, output_dir: str | Path | None = None) -> list[Path]:
    """Crop every detected box from a single image."""
    saved_paths: list[Path] = []
    for index, detection in enumerate(detections):
        box = [
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"],
        ]
        saved_paths.append(crop_box(image_path, box, output_dir=output_dir, index=index))
    return saved_paths
