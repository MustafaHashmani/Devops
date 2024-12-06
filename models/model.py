import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import os

def process_live_video(query_image_path, frame=None, top_n=5, output_dir="output", threshold=50, camera_index=0):
    # Load pre-trained ResNet and remove the last 3 layers
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-3]))
    model.eval()  # Set to evaluation mode

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the query image
    query_image = Image.open(query_image_path).convert("RGB")
    query_tensor = transform(query_image).unsqueeze(0)  # Add batch dimension

    # Extract feature vector for the query image
    with torch.no_grad():
        query_features = model(query_tensor).squeeze().numpy().flatten()  # Flatten to 1-D

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process single frame if provided
    if frame is not None:
        # Convert FileStorage to numpy array
        frame_array = np.frombuffer(frame.read(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = transform(frame_pil).unsqueeze(0)

        # Extract feature vector for the frame
        with torch.no_grad():
            frame_features = model(frame_tensor).squeeze().numpy().flatten()

        # Compute the cosine distance
        distance = cosine(query_features, frame_features)
        print("Cosine Distance:", distance)
        # Return the similarity score for the single frame
        return [{"similarity": float(1 - distance)}]

