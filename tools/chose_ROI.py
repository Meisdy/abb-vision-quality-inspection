"""
chose_ROI.py

Helper tool for interactively selecting and inspecting regions of interest (ROI) in an image
during development. Loads an image, lets the user select a rectangular ROI, shows result info,
and optionally saves the ROI as a PNG.
"""

import cv2

SAVE_ROI = False
DEFAULT_IMAGE_PATH = "../image_data/test_images/test_eval_correct_blue_00.jpg"
DEFAULT_OUTPUT_PATH = "roi_output.png"


def main(image_path: str = DEFAULT_IMAGE_PATH, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
    """
    Allow the user to select an ROI from a given image using OpenCV GUI.

    Args:
        image_path (str): Path to the source image.
        output_path (str): Where to save the cropped ROI image.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    print("Select ROI, press SPACE or ENTER to confirm, C to cancel")
    roi = cv2.selectROI("Select ROI", img, False)
    cv2.destroyWindow("Select ROI")

    if roi[2] == 0 or roi[3] == 0:
        print("No ROI selected")
        return

    x, y, w, h = [int(v) for v in roi]
    roi_img = img[y:y + h, x:x + w]

    print("\nROI Info:")
    print(f" Position: x={x}, y={y}")
    print(f" Size: width={w}, height={h}")
    print(f" Shape: {roi_img.shape}")

    if SAVE_ROI:
        success = cv2.imwrite(output_path, roi_img)
        print(f" Saved to: {output_path}{' (success)' if success else ' (FAILED!)'}")

    # Show the ROI
    cv2.imshow("Selected ROI", roi_img)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    image_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    output_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH
    main(image_arg, output_arg)
