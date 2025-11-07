import os
import numpy as np
import vision_pipeline
from vision_pipeline import load_images, Camera

# Configuration
FOLDER = 'validation'
input_folder = f'image_data/{FOLDER}'
output_folder = f'image_data/{FOLDER}/processed'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load raw images
images, filenames = load_images(input_folder)

# Process each image
for img, filename in zip(images, filenames):
    processed = Camera.preprocess(img)

    if processed is not None:
        # Save as .npy with same filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_res{vision_pipeline.IMAGE_SIZE}.npy")
        np.save(output_path, processed)
        print(f"✓ Saved {base_name}.npy")
    else:
        print(f"✗ Failed to process {filename}")

print(f"\nDone! Processed {len([f for f in images if f is not None])} images")
