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
import requests

app = Flask(__name__)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face URLs
HF_ALEXNET_URL = "https://huggingface.co/J7B/alexnet_model.pth/resolve/main/alexnet_model.pth"
HF_MOBILENET_URL = "https://huggingface.co/J7B/MobV2.h5/resolve/main/MobV2.h5"

# Paths
MODEL_DIR = "models"
ALEXNET_MODEL_PATH = os.path.join(MODEL_DIR, "alexnet_model.pth")
MOBILENET_MODEL_PATH = os.path.join(MODEL_DIR, "MobV2.h5")

# Auto-download models from Hugging Face
def download_models_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(ALEXNET_MODEL_PATH):
        print("📥 Downloading AlexNet model from Hugging Face...")
        response = requests.get(HF_ALEXNET_URL, stream=True)
        with open(ALEXNET_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ AlexNet download complete.")
    else:
        print("✔️ AlexNet model already exists locally.")

    if not os.path.exists(MOBILENET_MODEL_PATH):
        print("📥 Downloading MobileNetV2 model from Hugging Face...")
        response = requests.get(HF_MOBILENET_URL, stream=True)
        with open(MOBILENET_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ MobileNetV2 download complete.")
    else:
        print("✔️ MobileNetV2 model already exists locally.")

# Define the AlexNet model
class AlexNetMod(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetMod, self).__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

# Download models
download_models_if_missing()

# Load models
alexnet_model = AlexNetMod(num_classes=7).to(DEVICE)
alexnet_model.load_state_dict(torch.load(ALEXNET_MODEL_PATH, map_location=DEVICE))
alexnet_model.eval()

mobilenet_model = load_model(MOBILENET_MODEL_PATH)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class label mapping
idx_to_class = {
    0: "Actinic Keratosis & Intraepithelial Carcinoma (AKIEC)",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Benign Keratosis (BKL)",
    3: "Dermatofibroma (DF)",
    4: "Melanoma (MEL)",
    5: "Melanocytic Nevus (NV)",
    6: "Vascular Lesion (VASC)"
}

# Fusion model prediction
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

    # Normalize and fuse
    alexnet_probs /= np.sum(alexnet_probs)
    mobilenet_probs /= np.sum(mobilenet_probs)
    combined_probs = (alexnet_probs + mobilenet_probs) / 2
    final_prediction = np.argmax(combined_probs)

    return idx_to_class[final_prediction], combined_probs

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return render_template('index.html', result="No file uploaded.")

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template('index.html', result="Failed to load image.")

    try:
        predicted_class, combined_probs = soft_voting_fusion(image)
    except Exception as e:
        return render_template('index.html', result=f"Prediction error: {str(e)}")

    # Convert image to base64
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
