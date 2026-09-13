from pathlib import Path

# scripts/check_label.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "images"
LABEL_ROOT = PROJECT_ROOT / "labels"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

total_created = 0

for split in SPLITS:
    image_dir = IMAGE_ROOT / split
    label_dir = LABEL_ROOT / split

    label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"❌ Image folder not found: {image_dir}")
        continue

    created = 0
    existing = 0

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Cars0001.png -> Cars0001.txt
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            existing += 1
            continue

        # Create empty label
        label_path.touch()

        print(f"➕ Created: {label_path.name}")

        created += 1
        total_created += 1

    print(f"\n📁 {split}")
    print(f"   Existing: {existing}")
    print(f"   Created:  {created}")


print("\n--------------------------")
print(f"✅ Done!")
print(f"Created {total_created} empty labels.")
print("--------------------------")