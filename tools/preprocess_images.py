"""
preprocess_images.py

Batch-preprocesses images from a folder using the VisionProcessor, saving the results as .npy
files in an output directory. For use in dataset building, pipelines, or model pre-inference checks.
"""

import os
import numpy as np
from vision_pipeline import VisionProcessor

INPUT_FOLDER = r"C:\Users\Sandy\Desktop\temp"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "processed")
RESIZE_TO = 512


def main(input_folder: str = INPUT_FOLDER, output_folder: str = OUTPUT_FOLDER, resize_to: int = RESIZE_TO) -> None:
    """
    Preprocess images from a folder and save processed arrays as .npy files.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Destination for processed .npy files.
        resize_to (int): Target size for resizing.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load raw images and filenames
    images, filenames = VisionProcessor.load_images(input_folder)
    count_success = 0

    for img, filename in zip(images, filenames):
        processed = VisionProcessor.preprocess(img, resize_to=resize_to, visualisation=False)
        if processed is not None:
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}_res{resize_to}.npy")
            np.save(output_path, processed)
            print(f"✓ Saved {base_name}_res{resize_to}.npy")
            count_success += 1
        else:
            print(f"✗ Failed to process {filename}")

    print(f"\nDone! Processed {count_success} of {len(images)} images successfully.")


if __name__ == "__main__":
    main()
