# License Plate Detection with YOLO

A license plate detection project built with Python and Ultralytics YOLO.

The current model detects and locates license plates in vehicle images and live camera footage. The next stage of the project will add OCR to recognize the letters and numbers on detected plates.

## Project Structure

```text
license-plate/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/
│   ├── camera.py
│   ├── check_label.py
│   ├── evaluate.py
│   ├── fine-tune.py
│   ├── predict.py
│   ├── split.py
│   └── train.py
│
├── best.pt
├── data.yaml
├── requirements.txt
├── yolo11n.pt
└── README.md
```

## Requirements

* Python 3.10+
* Ultralytics YOLO
* OpenCV

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Or install the main dependencies manually:

```bash
pip install ultralytics opencv-python
```

## Dataset Format

The dataset uses YOLO format.

Each image must have a matching `.txt` label:

```text
images/train/Cars0001.png
labels/train/Cars0001.txt
```

A label containing a license plate looks like:

```text
0 0.512 0.634 0.215 0.084
```

The format is:

```text
class_id x_center y_center width height
```

All coordinates are normalized between `0` and `1`.

The project currently has one class:

```text
0 = license_plate
```

If an image contains **no license plate**, its matching `.txt` file should be empty.

## Dataset Configuration

Configure the dataset in `data.yaml`.

Example:

```yaml
path: /absolute/path/to/license-plate

train: images/train
val: images/val
test: images/test

names:
  0: license_plate
```

Change `path` to the absolute path of the project on your computer.

## Check Missing Labels

The label checker scans the train, validation, and test datasets and creates missing empty label files.

Run:

```bash
python scripts/check_label.py
```

> Only use empty labels for images that genuinely contain no license plate.

## Split the Dataset

If you have a new unsplit dataset, run:

```bash
python scripts/split.py
```

The script splits the dataset into training, validation, and test sets.

Current split:

```text
Train: 66%
Validation: 22%
Test: 12%
```

## Train the Model

Start training with:

```bash
python scripts/train.py
```

The model starts from the pretrained YOLO weights:

```text
yolo11n.pt
```

Training results and model weights will be saved under the `runs/` directory.

The most useful weights are:

```text
best.pt
last.pt
```

`best.pt` is the checkpoint with the best validation performance.

## Evaluate the Model

Evaluate the trained model on the test dataset:

```bash
python scripts/evaluate.py
```

Important metrics include:

* **Precision** — how often detected plates are actually plates
* **Recall** — how many real plates the model successfully detects
* **mAP@50** — detection accuracy at IoU 0.50
* **mAP@50-95** — detection performance across stricter IoU thresholds

## Run Prediction

To detect a license plate in an image:

```bash
python scripts/predict.py
```

The script loads the trained model:

```text
best.pt
```

and predicts license plate bounding boxes.

## Run Live Camera Detection

To use the model with your webcam:

```bash
python scripts/camera.py
```

Press:

```text
q
```

to stop the camera.

## Fine-Tune the Model

After collecting additional difficult examples, fine-tune the existing model:

```bash
python scripts/fine-tune.py
```

Useful images for fine-tuning include:

* Small or distant license plates
* Angled plates
* Blurry plates
* Night and low-light images
* Bright reflections and glare
* Partially blocked plates
* Different vehicle types
* Different license plate designs
* Busy backgrounds
* Images where the model produces false detections
* Images where the model misses a real plate

When the model incorrectly detects something such as a logo, sticker, sign, or other rectangular object, add similar examples to help teach the model that those objects are not license plates.

## Current Model Performance

Current test results:

```text
Precision:   94.5%
Recall:      70.5%
mAP@50:      81.8%
mAP@50-95:   56.3%
```

The model has high precision, while improving recall is currently an important goal.

## Typical Workflow

```text
Collect Images
      ↓
Annotate License Plates
      ↓
Check Labels
      ↓
Split Dataset
      ↓
Train YOLO
      ↓
Evaluate
      ↓
Test on New Images
      ↓
Collect Difficult / Failed Examples
      ↓
Fine-Tune
      ↓
Evaluate Again
```

## Future Work

The current project focuses on **license plate detection**.

The next stage will be:

```text
Vehicle Image
      ↓
YOLO
      ↓
License Plate Detection
      ↓
Crop License Plate
      ↓
OCR
      ↓
Plate Number
```

Possible future improvements include:

* OCR for plate number recognition
* Better detection of small plates
* Night-time detection
* Video processing
* Real-time license plate recognition
* Saving detected plate numbers
* Avoiding duplicate detections across video frames

## Example

Input:

```text
Vehicle image
```

Output:

```text
License plate detected
Confidence: 0.94
Bounding box: [x1, y1, x2, y2]
```

The detected region can later be passed to an OCR model to produce:

```text
2AB1234
```

## Notes

This project is being developed as a learning project for computer vision, object detection, YOLO, and eventually OCR.
