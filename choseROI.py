import cv2
import numpy as np
import sys


def main():
    img = cv2.imread('image_data/test_131125_00.jpg')
    if img is None:
        print("Failed to read image")
        return

    print("Select ROI, press SPACE or ENTER to confirm, C to cancel")
    roi = cv2.selectROI("Select ROI", img, False)
    cv2.destroyWindow("Select ROI")

    if roi[2] == 0 or roi[3] == 0:
        print("No ROI selected")
        return

    x, y, w, h = [int(v) for v in roi]
    roi_img = img[y:y + h, x:x + w]

    # Save the ROI
    print(f"\nROI Info:")
    print(f"  Position: x={x}, y={y}")
    print(f"  Size: width={w}, height={h}")
    print(f"  Shape: {roi_img.shape}")
    print(f"  Saved to: roi_output.png")

    # Show it
    cv2.imshow("Selected ROI", roi_img)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
