from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

image_path = Path(r"C:\Users\DELL\Pictures\Screenshots\4.png")
checkpoint_path = Path("checkpoints/best_resnet18_final.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    # Use the same normalization as training/evaluation as the model expects the same input format.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["fake", "real"]

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model = model.to(device)
model.eval()

# Open the image manually because inference we one image path, not ImageFolder.
# convert("RGB") makes sure the image has 3 color channels, which ResNet expects.
image = Image.open(image_path).convert("RGB")

# Apply the same preprocessing used during training.
image_tensor = transform(image)

# The model expects a batch of images.
# A single image is [3, 224, 224], so we add a batch dimension: [3, 224, 224] -> [1, 3, 224, 224].
image_tensor = image_tensor.unsqueeze(0) # Add a new dimension at position 0.

image_tensor = image_tensor.to(device)

with torch.no_grad():
    # outputs shape (two values): [fake, real]
    outputs = model(image_tensor)

    # Convert raw scores (logits) into probability-like values.
    # dim=1 means apply softmax across the two class scores.
    probabilities = torch.softmax(outputs, 1)
    
    confidence, predicted_class = torch.max(probabilities, 1)

predicted_label = class_names[predicted_class.item()]
confidence_score = confidence.item()

fake_probability = probabilities[0][0].item()
real_probability = probabilities[0][1].item()

print("\nPrediction complete")
print("Image:", image_path)
print("Predicted class:", predicted_label)
print(f"Confidence: {confidence_score:.4f}")
print(f"Real probability: {real_probability:.4f}")
print(f"Fake probability: {fake_probability:.4f}")