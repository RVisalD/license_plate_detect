from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "best.pt"


class DetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image, optionally as a data URL")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    app.state.model = YOLO(str(MODEL_PATH))
    yield


app = FastAPI(
    title="License Plate Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "License plate detection API", "docs": "/docs"}


def _decode_image(encoded_image: str) -> np.ndarray:
    if encoded_image.startswith("data:"):
        _, separator, encoded_image = encoded_image.partition(",")
        if not separator:
            raise HTTPException(status_code=400, detail="Invalid image data URL")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 must contain valid base64 data") from None

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Base64 data does not contain a valid image")
    return image


def _predict(image: np.ndarray, confidence: float, http_request: Request) -> Any:
    results = http_request.app.state.model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    return results[0]


def _box_data(box: Any) -> dict[str, float]:
    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
    return {
        "confidence": float(box.conf[0]),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


@app.post("/detect")
def detect(request: DetectionRequest, http_request: Request) -> dict[str, Any]:
    image = _decode_image(request.image_base64)
    result = _predict(image, request.confidence, http_request)
    image_height, image_width = image.shape[:2]
    plate_crops = []
    for box in result.boxes:
        detection = _box_data(box)
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        left = max(0, min(int(x1), image_width))
        top = max(0, min(int(y1), image_height))
        right = max(0, min(int(x2), image_width))
        bottom = max(0, min(int(y2), image_height))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue

        crop_success, crop_output = cv2.imencode(".jpg", crop)
        if not crop_success:
            raise HTTPException(status_code=500, detail="Could not encode a detected plate crop")

        plate_crops.append(
            {
                **detection,
                "image_base64": base64.b64encode(crop_output).decode("ascii"),
                "image_media_type": "image/jpeg",
            }
        )

    return {"crops": plate_crops}