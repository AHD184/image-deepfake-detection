# Image Deepfake Detection System

This project detects whether a face image is real or deepfake using a fine-tuned deep learning model.

## Planned approach
- Dataset: FaceForensics++ for training, optional Celeb-DF for harder evaluation
- Model: EfficientNet-B0 with transfer learning in PyTorch
- Pipeline: frame extraction -> face cropping -> preprocessing -> training -> inference
- Demo: Gradio web app
- Deployment: Hugging Face Spaces

## Goal
Build a practical and well-documented deepfake detection pipeline that can be trained locally and deployed as a live demo.

## Project Structure

data/
    raw/        - original datasets
    interim/    - extracted frames
    processed/  - cropped faces for training

src/
    data/       - preprocessing scripts
    models/     - model definitions
    training/   - training pipeline
    inference/  - prediction pipeline

app/
    Gradio web demo
