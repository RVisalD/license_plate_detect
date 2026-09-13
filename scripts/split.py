from pathlib import Path
import random
import shutil

# Project root: license-plate/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source folders before splitting
SOURCE_IMAGE_DIR = PROJECT_ROOT / "images" / "images"
SOURCE_LABEL_DIR = PROJECT_ROOT / "images" / "labels"

# Output image folders
IMAGE_ROOT = PROJECT_ROOT / "images"

TRAIN_IMAGE_DIR = IMAGE_ROOT / "train"
VAL_IMAGE_DIR = IMAGE_ROOT / "val"
TEST_IMAGE_DIR = IMAGE_ROOT / "test"

# Output label folders
LABEL_ROOT = PROJECT_ROOT / "labels"

TRAIN_LABEL_DIR = LABEL_ROOT / "train"
VAL_LABEL_DIR = LABEL_ROOT / "val"
TEST_LABEL_DIR = LABEL_ROOT / "test"

# Split ratios
TRAIN_RATIO = 0.66
VAL_RATIO = 0.22
TEST_RATIO = 0.12

# Create output folders
for folder in (
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TEST_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
    TEST_LABEL_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

# Find source images
image_extensions = {".png", ".jpg", ".jpeg"}

images = sorted(
    image
    for image in SOURCE_IMAGE_DIR.iterdir()
    if image.suffix.lower() in image_extensions
)

if not images:
    raise FileNotFoundError(
        f"No images found in:\n{SOURCE_IMAGE_DIR}"
    )

print(f"Found {len(images)} images.")

# Shuffle reproducibly
random.seed(42)
random.shuffle(images)

# Calculate split sizes
total = len(images)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_images = images[:train_count]

val_images = images[
    train_count:train_count + val_count
]

test_images = images[
    train_count + val_count:
]

print()
print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")
print()


def move_dataset(images, image_destination, label_destination):

    for image_path in images:

        # Move image
        destination_image = image_destination / image_path.name

        shutil.move(
            str(image_path),
            str(destination_image)
        )

        # Find matching label
        label_path = SOURCE_LABEL_DIR / f"{image_path.stem}.txt"

        destination_label = (
            label_destination / f"{image_path.stem}.txt"
        )

        if label_path.exists():

            shutil.move(
                str(label_path),
                str(destination_label)
            )

        else:

            # No annotation = negative image
            # Create empty YOLO label
            destination_label.touch()

            print(
                f"Created empty label: "
                f"{destination_label.name}"
            )


# Move train set
move_dataset(
    train_images,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR
)

# Move validation set
move_dataset(
    val_images,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR
)

# Move test set
move_dataset(
    test_images,
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR
)

print()
print("✅ Dataset split complete!")

print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")
print(f"Test:  {len(test_images)} images")