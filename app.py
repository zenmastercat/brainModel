import os
import io
import base64
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import urllib.request

# Import your model classes from the other file
from model_definitions import SimpleCNN, SimpleUNet

# Initialize the Flask app
app = Flask(__name__)

# --- MODEL AND TRANSFORMS LOADING ---
device = torch.device("cpu")

# --- URLS for your hosted models ---
# IMPORTANT: Replace these with your actual direct download links
CLS_MODEL_URL = "https://huggingface.co/zenmastercat/brain-mri-analyzer-models/resolve/main/classification_model_quantized.pth"
SEG_MODEL_URL = "https://huggingface.co/zenmastercat/brain-mri-analyzer-models/resolve/main/segmentation_model_weights.pth"

# Define local paths to save the models
MODELS_DIR = "models"
CLS_MODEL_PATH = os.path.join(MODELS_DIR, "classification_model_quantized.pth")
SEG_MODEL_PATH = os.path.join(MODELS_DIR, "segmentation_model_weights.pth")

def download_model(url, path):
    """Downloads a file from a URL to a given path."""
    if not os.path.exists(path):
        print(f"Downloading model from {url} to {path}...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise e

# Download models on startup
download_model(CLS_MODEL_URL, CLS_MODEL_PATH)
download_model(SEG_MODEL_URL, SEG_MODEL_PATH)


# Define class names for classification
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the image transforms
cls_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

seg_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the trained models
def load_models():
    """Load and return the trained models from local files."""
    # --- Load Quantized Classification Model ---
    cls_model_orig = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
    cls_model_quantized = torch.quantization.quantize_dynamic(
        cls_model_orig, {torch.nn.Linear}, dtype=torch.qint8
    )
    cls_model_quantized.load_state_dict(
        torch.load(CLS_MODEL_PATH, map_location=device, weights_only=False)
    )
    cls_model_quantized.eval()

    # --- Load Segmentation Model ---
    seg_model = SimpleUNet().to(device)
    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device, weights_only=False))
    seg_model.eval()

    return cls_model_quantized, seg_model

cls_model, seg_model = load_models()
print("✅ Models loaded successfully and in evaluation mode.")

# --- HELPER FUNCTIONS ---
def process_image(image_pil, transform):
    """Process a PIL image using the given transform."""
    return transform(image_pil).unsqueeze(0).to(device)

def get_tumor_info(mask_tensor):
    """Calculates the relative size of the tumor and creates a base64 mask image."""
    # Ensure tensor is on CPU and squeezed
    mask_tensor = mask_tensor.squeeze().cpu()
    
    # Apply sigmoid to get probabilities and threshold to get a binary mask
    binary_mask = torch.sigmoid(mask_tensor) > 0.5
    
    # Calculate relative size
    total_pixels = binary_mask.nelement()
    tumor_pixels = binary_mask.sum().item()
    relative_size = (tumor_pixels / total_pixels) if total_pixels > 0 else 0
    
    # Convert binary mask to an image for visualization
    mask_image = transforms.ToPILImage()(binary_mask.float())
    
    # Save image to a byte buffer
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    
    # Encode buffer to base64
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return mask_base64, relative_size

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the image upload and model prediction."""
    if 'file' not in request.json:
        return jsonify({'error': 'No file part'}), 400
    
    image_data = base64.b64decode(request.json['file'])
    image_pil = Image.open(io.BytesIO(image_data))

    # --- Classification ---
    image_rgb = image_pil.convert("RGB")
    cls_input = process_image(image_rgb, cls_transform)
    with torch.no_grad():
        cls_output = cls_model(cls_input)
        probabilities = torch.nn.functional.softmax(cls_output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_percent = f"{confidence.item()*100:.2f}%"

    # --- Segmentation ---
    image_gray = image_pil.convert("L")
    seg_input = process_image(image_gray, seg_transform)
    with torch.no_grad():
        seg_output = seg_model(seg_input)
    
    # Get mask image and tumor size
    mask_base64, tumor_size_ratio = get_tumor_info(seg_output)

    return jsonify({
        'classification': predicted_class,
        'confidence': confidence_percent,
        'mask': mask_base64,
        'tumor_size': tumor_size_ratio
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
