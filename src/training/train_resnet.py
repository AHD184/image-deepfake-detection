# We are using PyTorch to load our processed face dataset.
# Later, these images will be passed into ResNet, which is a pretrained CNN model.
# CNN = Convolutional Neural Network, commonly used for image classification.
# ResNet = Residual Network, a strong pretrained CNN architecture.

from pathlib import Path
import torch
import torch.nn as nn  # We can work on the individual parts of the neural network
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Training images are stored here:
train_dir = Path("data/processed/final/train")

# Validation images are stored here:
val_dir = Path("data/processed/final/val")

# transform is a pipeline of preprocessing steps.
# Every training image will go through these steps before being given to the model.
train_transform = transforms.Compose([

    # ResNet expects input images of size 224 x 224, so we scale the image accordingly.
    transforms.Resize((224, 224)),

    # Randomly flips some images left/right during training.
    # This helps the model learn face patterns instead of memorizing exact image layouts.
    transforms.RandomHorizontalFlip(p=0.5),

    # Makes the model less dependent on exact brightness, contrast, and sharpness.
    # This is useful because earlier the model seemed to rely too much on image quality.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    # Small blur sometimes, so sharpness alone does not decide fake/real.
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.2),

    # Convert the image into the tensor format PyTorch expects.
    # Shape becomes [3, 224, 224]: RGB channels, height, width.
    transforms.ToTensor(),

    # Normalizes the image using ImageNet mean and standard deviation.
    # This is needed because pretrained ResNet was trained with this normalization.
    # The three values are for the Red, Green, and Blue channels.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation should stay clean.
# We do not use random augmentation here because validation should measure
# how the model performs on the actual images.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ImageFolder automatically reads images from subfolders.
# PyTorch assigns class labels alphabetically:
# fake -> 0, real -> 1
train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

val_dataset = ImageFolder(
    root=val_dir,
    transform=val_transform
)


# DataLoader gives images to the model in batches.
# shuffle=True is used for training so images are mixed.
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Expected output: {'fake': 0, 'real': 1}
print("Class mapping:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Get one batch from the training DataLoader.
# iter(train_loader) prepares the loader to give batches one by one.
# next(...) takes the first batch.
# images contains a batch of image tensors.
# labels contains the correct label: 0 for fake and 1 for real.
images, labels = next(iter(train_loader))

print("\nOne batch loaded successfully")

# torch.Size([64, 3, 224, 224])
# 64 = number of images in this batch, 3 = RGB channels, 224 = height, 224 = width
print("Image batch shape:", images.shape)

# torch.Size([64])
print("Label batch shape:", labels.shape)
print("Example labels:", labels[:10])

# Use GPU if available, otherwise use CPU.
# GPU is much faster for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# Load ResNet18 with pretrained ImageNet weights.
# This gives us the correct ResNet18 structure before loading our Kaggle checkpoint.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ResNet18 was originally trained to classify 1000 ImageNet classes.
# We only have 2 classes, so we replace the final layer with a new 2-output layer.
num_features = model.fc.in_features      # Usually 512 for ResNet18
model.fc = nn.Linear(num_features, 2)    # 512 features -> 2 class scores

# Start from the Kaggle-trained model instead of starting only from ImageNet.
# The Kaggle model already learned fake/real face patterns, and HiDF will fine-tune it.
starting_checkpoint = Path("checkpoints/best_resnet18_kaggle.pth")
model.load_state_dict(torch.load(starting_checkpoint, map_location=device))

# Move the model to the selected device.
# Images and labels must also be moved to this same device during training.
model = model.to(device)

print("\nResNet18 model loaded successfully")
print("Starting checkpoint:", starting_checkpoint)
print("Final layer:")
print(model.fc)

criterion = nn.CrossEntropyLoss()  # Measures how wrong the model is for fake/real classification

# We use a smaller learning rate because this is fine-tuning.
# 0.00003 is strong enough to adapt to HiDF, but not as aggressive as 0.0001.
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)

num_epochs = 5  # One epoch means the model sees the full training dataset once.
best_val_acc = 0.0

# Save this as a separate model so we do not overwrite the Kaggle-only or clean HiDF models.
checkpoint_path = Path("checkpoints/best_resnet18_final.pth")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 30)

    model.train()  # Enables training mode, so weights can be updated.

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:  # Loop through data batch by batch
        # The model, images, and labels must all be on the same device.
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear gradients from the previous batch, not done automatically

        outputs = model(images)  # Shape: [batch_size, 2]
        # Ex: outputs = tensor([[2.5, 0.8], [0.4, 1.9]])
        # index 0 is fake and index 1 is real

        loss = criterion(outputs, labels)

        loss.backward()   # Calculate gradients for the model weights.
        optimizer.step()  # Update the weights using those gradients.

        # CrossEntropyLoss gives average loss for the batch, so multiply by batch size
        # before adding it to the epoch total.
        train_loss += loss.item() * images.size(0)
        # loss.item() converts the loss tensor into a normal Python number.
        # images.size(0) gives the number of images in this batch.

        # Choose the class with the higher score, but we do not need the score value.
        # Dimension 1 means we compare across the two class scores.
        _, predicted = torch.max(outputs, 1)  # Ex: predicted = tensor([0, 1, 1, 0])

        train_correct += (predicted == labels).sum().item()
        # Example output of (predicted == labels).sum() is tensor(3),
        # and .item() converts it to a normal Python number.

        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    model.eval()  # Evaluation mode: the model predicts, but training behavior is disabled.

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Validation does not update weights, so gradients are not needed.
    # This saves memory and makes validation faster.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val Accuracy:   {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)

        print(f"Best model saved with Val Accuracy: {best_val_acc:.4f}")

print("\nTraining complete")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best model saved at: {checkpoint_path}")