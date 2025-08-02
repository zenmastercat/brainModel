import pydicom
import nibabel as nib
import numpy as np
from skimage import measure
from skimage.transform import resize
import torch
import os

def load_dicom_series(directory):
    """Reads a directory of DICOM files, sorts them, and stacks them into a 3D volume."""
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    # Sort files based on the 'InstanceNumber' or 'ImagePositionPatient' DICOM tag
    files.sort(key=lambda x: int(x.InstanceNumber))
    
    # Stack the pixel arrays to create a 3D numpy array
    image_stack = np.stack([f.pixel_array for f in files])
    return image_stack.astype(np.float32)

def load_nifti_file(file_path):
    """Reads a NIfTI file and returns the 3D image data as a numpy array."""
    img = nib.load(file_path)
    return img.get_fdata().astype(np.float32)

def normalize_volume(volume):
    """Normalize the volume to be between 0 and 1."""
    if np.max(volume) > np.min(volume):
        volume -= np.min(volume)
        volume /= np.max(volume)
    return volume

def run_3d_segmentation_model(volume, model, device, patch_size=(128, 128, 128)):
    """
    Runs the 3D segmentation model on the volume.
    """
    print("Running 3D segmentation model...")
    # Preprocess the volume for the model
    original_shape = volume.shape
    volume_resized = resize(volume, patch_size, anti_aliasing=True)
    
    # Add batch and channel dimensions
    input_tensor = torch.from_numpy(volume_resized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the output
    # Remove batch and channel dimensions and move to CPU
    mask_resized = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
    
    # Resize the mask back to the original volume size
    mask = resize(mask_resized, original_shape, anti_aliasing=False, order=0) # order=0 for nearest neighbor
    
    return mask.astype(np.uint8)

def create_3d_mesh(mask, level=0.5):
    """
    Creates a 3D mesh from a binary mask using the marching cubes algorithm.
    Returns vertices and faces.
    """
    if np.max(mask) == 0:
        return None, None # No tumor found

    # Use scikit-image's marching cubes to find surfaces
    verts, faces, _, _ = measure.marching_cubes(mask, level)
    return verts.tolist(), faces.tolist()
