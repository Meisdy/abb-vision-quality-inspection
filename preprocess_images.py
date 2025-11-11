import os
import numpy as np
from vision_pipeline import VisionProcessor

# Configuration
input_folder = f'image_data/Correct_Yellow'
output_folder = input_folder + '/processed'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load raw images
images, filenames = VisionProcessor.load_images(input_folder)

# Process each image
for img, filename in zip(images, filenames):
    processed = VisionProcessor.preprocess(img, resize_to=512, visualisation=False)

    if processed is not None:
        # Save as .npy with same filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_res512.npy")
        np.save(output_path, processed)
        print(f"✓ Saved {base_name}.npy")
    else:
        print(f"✗ Failed to process {filename}")

print(f"\nDone! Processed {len([f for f in images if f is not None])} images")
