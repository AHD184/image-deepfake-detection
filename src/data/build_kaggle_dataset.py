from pathlib import Path
import shutil

kaggle_dir = Path("data/external/kaggle")
processed_dir = Path("data/processed")

folders = [
    processed_dir / "train" / "fake",
    processed_dir / "train" / "real",
    processed_dir / "val" / "fake",
    processed_dir / "val" / "real"
]

for folder in folders:
    # Remove the old processed dataset first.
    # rmtree deletes the whole folder, including all images inside it.
    # This prevents the old FF++ images from mixing with the new Kaggle dataset.
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)

def copy_images(input_dir, output_dir):
    # This function copies all JPG images from one folder to another.
    # input_dir is the original Kaggle folder.
    images = list(input_dir.glob("*.jpg"))

    for img in images:
        # copy2 preserves extra file information like modified time, used only for safety 
        # img.name keeps only the filename, ex: fake_001.jpg
        shutil.copy2(img, output_dir / img.name)

    print(f"{input_dir} -> {output_dir}: {len(images)} images copied")

copy_images(
    kaggle_dir / "Train" / "Fake",
    processed_dir / "train" / "fake"
)

copy_images(
    kaggle_dir / "Train" / "Real",
    processed_dir / "train" / "real"
)

copy_images(
    kaggle_dir / "Validation" / "Fake",
    processed_dir / "val" / "fake"
)

copy_images(
    kaggle_dir / "Validation" / "Real",
    processed_dir / "val" / "real"
)

print("\nKaggle dataset copied into data/processed")
print("Label mapping will remain:")
print("Fake: 0")
print("Real: 1")