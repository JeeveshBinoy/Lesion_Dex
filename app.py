from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms, models

app = Flask(__name__)

# Define the AlexNet model (PyTorch)
class AlexNetMod(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetMod, self).__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # Modify last layer for 7 classes

    def forward(self, x):
        return self.model(x)

# Load models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALEXNET_MODEL_PATH = "C:\\Users\\Nehal\\OneDrive\\Desktop\\LesionDex\\Lesion_Dex\\models\\alexnet_model.pth"
MOBILENET_MODEL_PATH = "C:\\Users\\Nehal\\OneDrive\\Desktop\\LesionDex\\Lesion_Dex\\models\\MobV2.h5"

alexnet_model = AlexNetMod(num_classes=7).to(DEVICE)
alexnet_model.load_state_dict(torch.load(ALEXNET_MODEL_PATH))
alexnet_model.eval()

mobilenet_model = load_model(MOBILENET_MODEL_PATH)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class mapping with full names
idx_to_class = {
    0: "Actinic Keratosis & Intraepithelial Carcinoma (AKIEC)",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Benign Keratosis (BKL)",
    3: "Dermatofibroma (DF)",
    4: "Melanoma (MEL)",
    5: "Melanocytic Nevus (NV)",
    6: "Vascular Lesion (VASC)"
}

# Soft Voting Fusion (Equal Weights)
def soft_voting_fusion(pil_img):
    image_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # AlexNet prediction
    with torch.no_grad():
        alexnet_outputs = alexnet_model(image_tensor)
        alexnet_probs = F.softmax(alexnet_outputs, dim=1).cpu().numpy()

    # MobileNetV2 prediction
    mobilenet_image = np.array(pil_img.resize((224, 224))) / 255.0
    mobilenet_image = np.expand_dims(mobilenet_image, axis=0)
    mobilenet_probs = mobilenet_model.predict(mobilenet_image)

    # Normalize probabilities
    alexnet_probs /= np.sum(alexnet_probs)
    mobilenet_probs /= np.sum(mobilenet_probs)

    # Soft Voting Fusion (Equal Weights)
    combined_probs = (alexnet_probs + mobilenet_probs) / 2
    final_prediction = np.argmax(combined_probs)

    return idx_to_class[final_prediction], combined_probs

# Home route for the upload page
@app.route('/')
def home():
    return render_template('index.html')

# Upload and predict route
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return render_template('index.html', result="No file uploaded.")
    
    # Read image
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return render_template('index.html', result="Failed to load image.")

    # Get prediction using soft voting fusion
    try:
        predicted_class, combined_probs = soft_voting_fusion(image)
    except Exception as e:
        return render_template('index.html', result="Prediction error.")

    # Convert image to base64 for inline HTML display
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template(
        'index.html',
        result=predicted_class,
        image_base64=image_base64
    )

if __name__ == "__main__":
    app.run(debug=True)
