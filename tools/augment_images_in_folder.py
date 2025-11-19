"""
augment_images_in_folder.py

Batch-augments images in a specified folder using lighting and color/saturation transformations.
Augmented images are saved with modified filenames alongside the originals. Useful for data
augmentation in ML pipelines or manual dataset enhancement.
"""

import os
import cv2
from vision_pipeline import VisionProcessor

FOLDER = r"C:\Users\Sandy\Desktop\temp"  # Point to the existing training images folder


def main(folder: str = FOLDER) -> None:
    """
    Augment all images in folder with lighting and color/saturation transforms,
    saving new images with recognizable filenames.

    Args:
        folder (str): Path to image folder to augment.
    """
    images, filenames = VisionProcessor.load_images(folder)
    print(f"Loaded {len(images)} images.")

    for img, fname in zip(images, filenames):
        # Augmentation 1 (lighting)
        img_aug1 = VisionProcessor.augment_lighting(img)
        fname_aug1 = f"{os.path.splitext(fname)[0]}_auglighting{os.path.splitext(fname)[1]}"
        out_path_aug1 = os.path.join(folder, fname_aug1)
        cv2.imwrite(out_path_aug1, img_aug1)

        # Augmentation 2 (color/saturation)
        img_aug2 = VisionProcessor.augment_color_saturation(img)
        fname_aug2 = f"{os.path.splitext(fname)[0]}_augcolor{os.path.splitext(fname)[1]}"
        out_path_aug2 = os.path.join(folder, fname_aug2)
        cv2.imwrite(out_path_aug2, img_aug2)

        print(f"Saved: {fname_aug1}, {fname_aug2}")

    print("Done! Augmented images saved alongside originals in the same folder.")


if __name__ == "__main__":
    import sys

    folder_arg = sys.argv[1] if len(sys.argv) > 1 else FOLDER
    main(folder_arg)
