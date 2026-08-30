from pathlib import Path
import random
import shutil

# Settings
SOURCE_DIR = Path("images")
IMAGE_DIR = SOURCE_DIR / "images"
LABEL_DIR = SOURCE_DIR / "labels"
TRAIN_DIR = SOURCE_DIR / "train"
VAL_DIR = SOURCE_DIR / "val"
TEST_DIR = SOURCE_DIR / "test"

# Split ratios
TRAIN_RATIO = 0.66
VAL_RATIO = 0.66
TEST_RATIO = 0.33

# Create output folders
for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Find source images
images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

print(f"Found {len(images)} images.")

# Shuffle to keep the split reproducible
random.seed(42)
random.shuffle(images)

# 66% for train, then split the remaining 34% into 66% val and 33% test
train_count = int(len(images) * TRAIN_RATIO)
train_images = images[:train_count]
remaining_images = images[train_count:]

if remaining_images:
    val_count = int(len(remaining_images) * (VAL_RATIO / (VAL_RATIO + TEST_RATIO)))
else:
    val_count = 0

val_images = remaining_images[:val_count]
test_images = remaining_images[val_count:]

# Move images and labels
for image in train_images:
    shutil.move(str(image), TRAIN_DIR / image.name)
    label = LABEL_DIR / f"{image.stem}.txt"
    if label.exists():
        shutil.move(str(label), TRAIN_DIR / label.name)

for image in val_images:
    shutil.move(str(image), VAL_DIR / image.name)
    label = LABEL_DIR / f"{image.stem}.txt"
    if label.exists():
        shutil.move(str(label), VAL_DIR / label.name)

for image in test_images:
    shutil.move(str(image), TEST_DIR / image.name)
    label = LABEL_DIR / f"{image.stem}.txt"
    if label.exists():
        shutil.move(str(label), TEST_DIR / label.name)

print("Done!")
print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")
print(f"Test:  {len(test_images)} images")