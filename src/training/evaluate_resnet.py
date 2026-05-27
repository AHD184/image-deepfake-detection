from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_dir = Path("data/processed/final/test")
checkpoint_path = Path("checkpoints/best_resnet18_final.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert image pixels into PyTorch tensor format

    # Use the same normalization as training because ResNet was pretrained on ImageNet.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ImageFolder reads the folder names as class labels.
# Since folders are alphabetical: fake = 0, real = 1.
test_dataset = ImageFolder(test_dir, transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False  # Keep test order fixed since we are only evaluating
)

print("Class mapping:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))

# Load the same ResNet18 structure that was used during training.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the original 1000-class ImageNet layer with our 2-class layer.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load the saved weights from the best training checkpoint.
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model = model.to(device)
model.eval()  # Evaluation mode: model predicts, but training behavior is disabled

correct = 0
total = 0

# Index 0 is fake and index 1 is real.
# These lists track performance for each class separately.
class_correct = [0, 0]
class_total = [0, 0]

# No gradients are needed because we are not training or updating weights here.
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # outputs shape: [batch_size, 2]
        # Each row has two scores: [fake_score, real_score].
        outputs = model(images)

        # Pick the class with the higher score for each image.
        _, predicted = torch.max(outputs, 1)

        # Count overall correct predictions.
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Count correct predictions for fake and real separately.
        for i in range(labels.size(0)):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()

            class_total[true_label] += 1

            if predicted_label == true_label:
                class_correct[true_label] += 1

accuracy = correct / total

print("\nEvaluation complete")
print(f"Overall test accuracy: {accuracy:.4f}")

# Convert {'fake': 0, 'real': 1} into {0: 'fake', 1: 'real'}
# so we can print readable class names below.
idx_to_class = {value: key for key, value in test_dataset.class_to_idx.items()}

for class_idx in range(len(class_correct)):
    class_name = idx_to_class[class_idx]

    if class_total[class_idx] > 0:
        class_accuracy = class_correct[class_idx] / class_total[class_idx]
    else:
        class_accuracy = 0

    print(
        f"{class_name}: "
        f"{class_correct[class_idx]}/{class_total[class_idx]} correct "
        f"({class_accuracy:.4f})"
    )