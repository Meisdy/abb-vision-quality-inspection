import os
import sys
import cv2
from vision_pipeline import Camera, VisionProcessor

os.environ["PYLON_CAMEMU"] = "3"

# Configuration
CONFIG_NAME = "test_eval_false"
NUM_ITERATIONS = 100
SAVE_DIR = "image_data/test_images"

USE_ROI = True  # now actively using the two ROIs
ROI_BOT = (575, 730, 1115, 381)  # (x, y, w, h)
ROI_TOP = (581, 90, 1110, 400)  # (x, y, w, h)


def draw_roi_overlay(image, rois, blackout=False, blur=True, blur_ksize=(35, 35)):
    """
    Visualize multiple ROIs: keep ROIs visible, mask the rest via black or blur.
    rois: list of (x, y, w, h)
    """
    if blackout:
        masked = image.copy()
        masked[:] = 0
        for (rx, ry, rw, rh) in rois:
            masked[ry:ry + rh, rx:rx + rw] = image[ry:ry + rh, rx:rx + rw]
        return masked
    elif blur:
        blurred = cv2.GaussianBlur(image, blur_ksize, 0)
        out = blurred
        for (rx, ry, rw, rh) in rois:
            out[ry:ry + rh, rx:rx + rw] = image[ry:ry + rh, rx:rx + rw]
        return out
    else:
        # default to black out
        return draw_roi_overlay(image, rois, blackout=True)


def capture_training_photos(camera, config_name, iteration, save_dir, cv_window):
    """
    SPACE: capture both ROI_TOP and ROI_BOT crops as separate JPGs.
    ESC: terminate program.
    Preview shows both ROIs, rest masked (black or blurred).
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Iteration {iteration + 1} ===")
    print("Press SPACE to capture, ESC to terminate")

    while True:
        image_raw = camera.capture_raw()
        if image_raw is None:
            print("ERROR: Failed to capture image")
            return False

        if USE_ROI:
            # Show both ROIs with the rest masked
            display_img = draw_roi_overlay(
                image_raw,
                rois=[ROI_TOP, ROI_BOT],
                blackout=True,  # set to False and blur=True to blur instead of black
                blur=False
            )
        else:
            display_img = image_raw.copy()

        # HUD
        cv2.putText(display_img, "Press SPACE to capture, ESC to terminate", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display_img, f"Iteration {iteration + 1}/{NUM_ITERATIONS}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Optional: draw visible rectangles around ROIs
        if USE_ROI:
            for color, roi in [((0, 255, 0), ROI_TOP), ((255, 0, 0), ROI_BOT)]:
                x, y, w, h = roi
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)

        cv2.imshow(cv_window, display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC - terminate
            return False

        if key == 32:  # SPACE - capture
            if USE_ROI:
                img_top = VisionProcessor.crop(image_raw, ROI_TOP)
                img_bot = VisionProcessor.crop(image_raw, ROI_BOT)

                # Save with explicit suffixes for clarity
                filename_top = f"{save_dir}/{config_name}_top_{iteration:02d}.jpg"
                filename_bot = f"{save_dir}/{config_name}_bot_{iteration:02d}.jpg"

                cv2.imwrite(filename_top, img_top)
                cv2.imwrite(filename_bot, img_bot)

                print(f"✓ Saved: {filename_top}")
                print(f"✓ Saved: {filename_bot}")
            else:
                filename_jpg = f"{save_dir}/{config_name}_{iteration:02d}.jpg"
                cv2.imwrite(filename_jpg, image_raw)
                print(f"✓ Saved: {filename_jpg}")

            return True


def main():
    """Main function"""
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Target: {NUM_ITERATIONS} captures")
    print(f"Save: {SAVE_DIR}\n")

    cv_window = 'Camera Feed'
    cv2.namedWindow(cv_window, cv2.WINDOW_AUTOSIZE)

    try:
        camera = Camera(exposure_time=30000.0, frame_rate=30.0)

        for iteration in range(NUM_ITERATIONS):
            success = capture_training_photos(camera, CONFIG_NAME, iteration, SAVE_DIR, cv_window)
            if not success:
                print("Terminated by user")
                break

        # Count both top/bot images
        jpg_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
        print(f"\n✓ Complete! Total captures (files): {jpg_count}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        camera.shutdown()


if __name__ == "__main__":
    main()
