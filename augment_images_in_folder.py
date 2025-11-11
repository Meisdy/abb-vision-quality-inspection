import os
import cv2
from vision_pipeline import VisionProcessor

FOLDER = r"image_data/Classifier/train/incorrect_mix"  # Point to the existing training images folder

images, filenames = VisionProcessor.load_images(FOLDER)
print(f"Loaded {len(images)} images.")

for img, fname in zip(images, filenames):
    # Augmentation 1 (lighting)
    img_aug1 = VisionProcessor.augment_lighting(img)
    fname_aug1 = f"{os.path.splitext(fname)[0]}_auglighting{os.path.splitext(fname)[1]}"
    out_path_aug1 = os.path.join(FOLDER, fname_aug1)
    cv2.imwrite(out_path_aug1, img_aug1)

    # Augmentation 2 (color/saturation)
    img_aug2 = VisionProcessor.augment_color_saturation(img)
    fname_aug2 = f"{os.path.splitext(fname)[0]}_augcolor{os.path.splitext(fname)[1]}"
    out_path_aug2 = os.path.join(FOLDER, fname_aug2)
    cv2.imwrite(out_path_aug2, img_aug2)

    print(f"Saved: {fname_aug1}, {fname_aug2}")

print("Done! Augmented images saved alongside originals in the same folder.")
