import torch
import numpy as np
from PIL import Image

# Assuming utils, models, losses are available from your project structure

def reconstruct_from_image(image_path, cfg, encoder, decoder):
    """
    Reconstructs a 3D voxel model from a single image.

    Args:
        image_path: Path to the input image (224x224).
        cfg: Configuration object.
        encoder: Trained encoder model.
        decoder: Trained decoder model.

    Returns:
        A NumPy array representing the reconstructed voxel model.
    """

    # 1. Image Loading and Preprocessing
    image = Image.open(image_path).convert('RGB')
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),  # Use CenterCrop for a single image. Remove RandomBackground.
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])
    rendering_images = test_transforms(image).unsqueeze(0)  # Add batch dimension.



    # 2. Inference
    device = torch.cuda.current_device()  # or 'cpu' if not using GPU
    rendering_images = rendering_images.to(device) # Move tensor to the same device as the model.

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        image_features = encoder(rendering_images)
        generated_volume = decoder(image_features).squeeze(dim=1)

    # 3. Postprocessing (Thresholding)
    threshold = 0.4  # Or use the best threshold from cfg if available.
    pred_volume = (generated_volume > threshold).cpu().numpy()

    return pred_volume


# Example usage:
# Assuming cfg, encoder, decoder are already loaded/initialized.
voxel_model = reconstruct_from_image("path/to/your/image.jpg", cfg, encoder, decoder)

# You can then visualize or save the voxel_model.  For saving as .binvox:
# from utils import binvox_rw
# with open("output.binvox", 'wb') as f:
#     vox = binvox_rw.Voxels(voxel_model[0], (32,) * 3, (0,) * 3, 1, 'xzy') # Assuming output size 32x32x32. Adjust as needed.
#     vox.write(f)

