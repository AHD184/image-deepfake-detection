# We create two datasets from the faces extracted
# They will be split 80/20 
# One will be used for training and the other will be used for validating the output

from pathlib import Path
import random # Shuffles data
import shutil # Used to copy files from one folder to another

# We set a random seed which make the shuffle reproducible
# Running the code again will give the same split 
random.seed(42)

real_dir = Path("data/interim/real_faces")
fake_dir = Path("data/interim/fake_faces")

base_dir = Path("data/processed")

train_real_dir = base_dir / "train" / "real"
train_fake_dir = base_dir / "train" / "fake"
val_real_dir = base_dir / "val" / "real"
val_fake_dir = base_dir / "val" / "fake"

for folder in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
    folder.mkdir(parents=True, exist_ok=True)


# This function returns a list of paths of all the faces from each file
def collect_images(input_dir):
    return sorted(input_dir.glob("*/*.jpg"))

# This function copies all images into destination_dir using unique filenames
# Saved with unique filenames because there may be multiple frames with the same name across files
# Example: frame_0000.jpg -> 033_frame_0000.jpg
def copy_images(image_paths, destination_dir):
     for image_path in image_paths:
         new_name = f"{image_path.parent.name}_{image_path.name}"

         destination_path = destination_dir / new_name
         shutil.copy(image_path, destination_path)

def process_class(input_dir, train_dir, val_dir, split_ratio=0.8):
    image_paths = collect_images(input_dir)

    if len(image_paths) == 0: # List
        print(f"No images found in: {input_dir}")
        return
    
    # Shuffle each frame so that all frames from one particular video don't end up in one dataset only
    random.shuffle(image_paths) 
    
    # Number of images multiplied with the ratio
    split_index = int(len(image_paths) * split_ratio)

    train_images = image_paths[:split_index]
    val_images = image_paths[split_index:]

    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)

    print(f"\nClass: {input_dir.name}")
    print(f"Total images: {len(image_paths)}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

process_class(real_dir, train_real_dir, val_real_dir, split_ratio=0.8)
process_class(fake_dir, train_fake_dir, val_fake_dir, split_ratio=0.8)

print("\nDataset building complete.")