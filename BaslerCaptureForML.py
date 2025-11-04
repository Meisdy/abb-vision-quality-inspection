import os
import sys
import cv2
import numpy as np
import time
from vision_pipeline import Camera

os.environ["PYLON_CAMEMU"] = "3"

# Configuration
CONFIG_NAME = "correct_assembly_TEST"
NUM_ITERATIONS = 2
SAVE_DIR = "image_data/train"


def capture_training_photos(camera, config_name, iteration, save_dir):
    """
    Captures 2 photos, saves both .npy (for training) and .jpg (for viewing)
    Press SPACE to start capture, ESC to cancel
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Iteration {iteration + 1} ===")
    print("Press SPACE to capture 2 photos, ESC to cancel")

    cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)

    waiting_for_trigger = True
    photos_captured = 0

    while True:
        # Capture raw image
        image_raw = camera.capture_raw()

        if image_raw is None:
            print("ERROR: Failed to capture image")
            cv2.destroyAllWindows()
            return False

        # Display with instructions
        display_img = image_raw.copy()

        if waiting_for_trigger:
            text = "Press SPACE to capture"
            cv2.putText(display_img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        else:
            text = f"Capturing... Photo {photos_captured + 1}/2"
            cv2.putText(display_img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

        cv2.imshow('Camera Feed', display_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Cancelled by user")
            cv2.destroyAllWindows()
            return False

        elif key == 32 and waiting_for_trigger:  # SPACE to start
            waiting_for_trigger = False
            photos_captured = 0
            print("Starting capture sequence...")

        # Auto-capture 2 photos with 2s delay
        if not waiting_for_trigger and photos_captured < 2:
            if photos_captured == 0:
                # Preprocess
                image_processed = camera.preprocess(image_raw)

                # Save both .npy (for training) and .jpg (for viewing)
                filename_npy = f"{save_dir}/{config_name}_{iteration:02d}_01.npy"
                filename_jpg = f"{save_dir}/{config_name}_{iteration:02d}_01.jpg"

                np.save(filename_npy, image_processed)
                cv2.imwrite(filename_jpg, image_raw)

                print(f"✓ Photo 1/2 saved")
                print(f"  - Processed: {filename_npy}")
                print(f"  - Viewable: {filename_jpg}")
                photos_captured += 1

                # Wait 2 seconds before next capture
                time.sleep(2)

            elif photos_captured == 1:
                # Preprocess
                image_processed = camera.preprocess(image_raw)

                # Save both .npy (for training) and .jpg (for viewing)
                filename_npy = f"{save_dir}/{config_name}_{iteration:02d}_02.npy"
                filename_jpg = f"{save_dir}/{config_name}_{iteration:02d}_02.jpg"

                np.save(filename_npy, image_processed)
                cv2.imwrite(filename_jpg, image_raw)

                print(f"✓ Photo 2/2 saved")
                print(f"  - Processed: {filename_npy}")
                print(f"  - Viewable: {filename_jpg}")
                photos_captured += 1

                # Done with this iteration
                cv2.destroyAllWindows()
                return True


def main():
    """Main function to run capture session"""
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Target iterations: {NUM_ITERATIONS}")
    print(f"Save location: {SAVE_DIR}")

    try:
        # Initialize camera once
        camera = Camera(exposure_time=30000.0, frame_rate=30.0)

        # Capture loop
        for iteration in range(NUM_ITERATIONS):
            print(f"\n{'=' * 50}")
            print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
            print(f"{'=' * 50}")

            if iteration == 0:
                input("Place assembly, press Enter...")
            else:
                input("Replace assembly, press Enter...")

            success = capture_training_photos(
                camera,
                CONFIG_NAME,
                iteration,
                SAVE_DIR
            )

            if not success:
                print("Session cancelled by user")
                break

        print(f"\n✓ Capture session complete!")
        npy_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.npy')])
        jpg_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
        print(f"Total processed (.npy): {npy_count}")
        print(f"Total viewable (.jpg): {jpg_count}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        camera.shutdown()


if __name__ == "__main__":
    main()
