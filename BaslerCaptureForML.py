import os
import sys
import cv2
import vision_pipeline
from vision_pipeline import Camera

os.environ["PYLON_CAMEMU"] = "3"

# Configuration
CONFIG_NAME = "false_yellow_071125"
NUM_ITERATIONS = 50
SAVE_DIR = "image_data/validation"


def capture_training_photos(camera, config_name, iteration, save_dir, cv_window):
    """
    Single SPACE press to capture one photo per iteration.
    ESC terminates program.
    Saves only JPG (raw images).
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Iteration {iteration + 1} ===")
    print("Press SPACE to capture, ESC to terminate")

    while True:
        # Capture raw image
        image_raw = camera.capture_raw()

        if image_raw is None:
            print("ERROR: Failed to capture image")
            return False

        # Display
        display_img = image_raw.copy()
        cv2.putText(display_img, "Press SPACE to capture, ESC to terminate", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display_img, f"Iteration {iteration + 1}/{NUM_ITERATIONS}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow(cv_window, display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC - terminate
            return False

        if key == 32:  # SPACE - capture
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

        jpg_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
        print(f"\n✓ Complete! Total captures: {jpg_count}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        camera.shutdown()


if __name__ == "__main__":
    main()
