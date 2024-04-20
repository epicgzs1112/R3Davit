from PIL import Image
import os

# Path to the folder containing images
folder_path = "/root/autodl-tmp/R3DSWIN++/pix3d/img/chair"

# Desired dimensions
new_width = 224
new_height = 244

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a PNG image
    if filename.endswith(".png"):
        # Open the image
        with Image.open(os.path.join(folder_path, filename)) as img:
            # Resize the image
            resized_img = img.resize((new_width, new_height))
            # Replace the original image with the resized one
            resized_img.save(os.path.join(folder_path, filename))
