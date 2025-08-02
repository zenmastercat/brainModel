import os
import io
import base64
import zipfile
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch

# Import helper functions from our new utils file
import utils 

# Import the 3D model definition
# You will need a model_definitions.py file with the Simple3DUNet class
from model_definitions import Simple3DUNet 

# Initialize the Flask app
app = Flask(__name__)

# --- MODEL LOADING ---
device = torch.device("cpu")

# Define the local path for the 3D model
MODELS_DIR = "models"
SEG_3D_MODEL_PATH = os.path.join(MODELS_DIR, "3d_segmentation_model_weights.pth")

def load_3d_model():
    """Load and return the trained 3D segmentation model."""
    # NOTE: You will need to train the 3D model and place the weights file
    # in a 'models' directory for this to work.
    if not os.path.exists(SEG_3D_MODEL_PATH):
        print("WARNING: 3D model weights not found. Using a placeholder for segmentation.")
        return None

    model = Simple3DUNet().to(device)
    # Ensure you have a model_definitions.py file with the Simple3DUNet class
    # You might need to add 'weights_only=False' depending on your PyTorch version
    model.load_state_dict(torch.load(SEG_3D_MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 3D Segmentation model loaded successfully.")
    return model

seg_3d_model = load_3d_model()


# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict_3d', methods=['POST'])
def predict_3d():
    """Handle the 3D file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create a temporary directory to store uploaded files
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    try:
        # --- Load 3D Volume ---
        if file.filename.endswith('.nii') or file.filename.endswith('.nii.gz'):
            volume = utils.load_nifti_file(file_path)
        elif file.filename.endswith('.zip'):
            # Unzip the file and treat it as a DICOM series
            dicom_dir = os.path.join(temp_dir, "dicom_series")
            os.makedirs(dicom_dir, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dicom_dir)
            volume = utils.load_dicom_series(dicom_dir)
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a .nii, .nii.gz, or a .zip of DICOM files.'}), 400

        # --- Process Volume ---
        # Transpose to (Depth, Height, Width) if it's not already
        if volume.shape[2] < volume.shape[0] and volume.shape[2] < volume.shape[1]:
             volume = np.transpose(volume, (2, 1, 0))

        # Normalize the volume for better visualization
        volume = utils.normalize_volume(volume)
        
        # --- Run Segmentation ---
        if seg_3d_model:
            mask_3d = utils.run_3d_segmentation_model(volume, seg_3d_model, device)
        else:
            # Fallback to placeholder if the real model isn't loaded
            # This placeholder is defined in your utils.py
            # To use it, you'll need to re-add the placeholder logic to utils.py
            # For now, we assume a real model or an error.
            return jsonify({'error': '3D segmentation model is not loaded.'}), 500


        # --- Generate 3D Tumor Mesh ---
        verts, faces = utils.create_3d_mesh(mask_3d)
        
        # --- Prepare Slices for 2D Viewer ---
        # Convert each slice to a base64 encoded PNG
        slices_base64 = []
        for i in range(volume.shape[0]):
            slice_img = Image.fromarray((volume[i] * 255).astype(np.uint8), 'L')
            buffered = io.BytesIO()
            slice_img.save(buffered, format="PNG")
            slices_base64.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            
        mask_slices_base64 = []
        for i in range(mask_3d.shape[0]):
            mask_slice_img = Image.fromarray(mask_3d[i] * 255, 'L')
            buffered = io.BytesIO()
            mask_slice_img.save(buffered, format="PNG")
            mask_slices_base64.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Failed to process file. Error: {e}'}), 500
    
    # --- Return Results ---
    return jsonify({
        'brain_slices': slices_base64,
        'mask_slices': mask_slices_base64,
        'tumor_mesh': {
            'vertices': verts,
            'faces': faces
        } if verts else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
