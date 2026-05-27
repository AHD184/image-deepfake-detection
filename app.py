from pathlib import Path
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = Path("best_resnet18_final.pth")

class_names = ["fake", "real"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model

model = load_model()

def predict(image):
    if image is None:
        return "Please upload an image."
    
    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        fake_probability = probabilities[0][0].item()
        real_probability = probabilities[0][1].item()

    if fake_probability > real_probability:
        predicted_label = "Fake"
        confidence = fake_probability
    else:
        predicted_label = "Real"
        confidence = real_probability

    # If the model is not confident enough, avoid forcing a fake/real answer.
    # This makes the app more honest for borderline cases.
    confidence_threshold = 0.75

    if confidence < confidence_threshold:
        final_prediction = "Uncertain"
    else:
        final_prediction = predicted_label

    result_text = (
        f"Prediction: {predicted_label}\n"
        f"Confidence: {confidence:.2%}\n\n"
        f"Fake probability: {fake_probability:.2%}\n"
        f"Real probability: {real_probability:.2%}"
)
    scores = {
        "Fake": fake_probability,
        "Real": real_probability
    }

    return result_text, scores

# Create the Gradio interface.
# This connects the upload box to our predict() function.
demo = gr.Interface(
    fn=predict, # Function that runs when the user uploads an image

    # Gradio will pass the uploaded image to predict() as a PIL image.
    inputs = gr.Image(type="pil", label="Upload a face image"),

    # The app will show two outputs:
    # 1. Text summary with prediction and confidence
    # 2. Class probability labels for Fake and Real
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities")
    ],
    
    title = "Image Deepfake Detection System",
    # Short note shown on the app page.
    # This is important because the model works best on cropped face images.
    description=(
        "Instructions: upload a clear cropped face image, then the app will return Fake, Real, or Uncertain with class probabilities. "
        "\nLimitations: this is a demo-level model. It may be affected by lighting, compression, makeup, editing, screenshots, and dataset bias. It is not designed to reliably detect the latest high-quality deepfakes, which usually require larger datasets, stronger architectures, video-based analysis, and more advanced detection methods."
    ),
    flagging_mode="never"
)

# Only launch the app when this file is run directly.
# This prevents the app from starting automatically if the file is imported elsewhere.
if __name__ == "__main__":
    demo.launch()