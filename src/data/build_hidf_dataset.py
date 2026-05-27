from pathlib import Path
import random
import shutil


# Fix the shuffle so the train/val/test split is the same every time we run this script.
# The value 42 is just a common seed value; any fixed number would work.
random.seed(42)


hidf_fake_dir = Path("data/external/hidf/fake")
hidf_real_dir = Path("data/external/hidf/real")

# HiDF will be stored in a separate processed folder so it does not overwrite the Kaggle processed dataset.
processed_dir = Path("data/processed/final")


output_folders = [
    processed_dir / "train" / "fake",
    processed_dir / "train" / "real",
    processed_dir / "val" / "fake",
    processed_dir / "val" / "real",
    processed_dir / "test" / "fake",
    processed_dir / "test" / "real"
]


for folder in output_folders:
    # Create all required output folders before copying images into them.
    folder.mkdir(parents=True, exist_ok=True)


def get_images(input_dir):
    # HiDF has both JPG and PNG files, so we collect the common image formats.
    images = []
    images.extend(input_dir.glob("*.jpg"))
    images.extend(input_dir.glob("*.jpeg"))
    images.extend(input_dir.glob("*.png"))

    return images


def split_and_copy(input_dir, class_name):
    images = get_images(input_dir)

    # Shuffle before splitting so train/val/test get a random mix of images.
    # Because of random.seed(42), this shuffle is still reproducible.
    random.shuffle(images)

    total = len(images)

    # 70% train, 15% validation, 15% test.
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # .items() gives both the split name and its image list, not only the key.
    for split_name, split_images in splits.items():
        output_dir = processed_dir / split_name / class_name

        for img in split_images:
            # copy2 copies the file and keeps basic file metadata.
            shutil.copy2(img, output_dir / img.name)

    print(f"\n{class_name.upper()} images:")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")


split_and_copy(hidf_fake_dir, "fake")
split_and_copy(hidf_real_dir, "real")


print("\nHiDF dataset ready in data/processed/final")
print("Label mapping will remain:")
print("fake = 0")
print("real = 1")