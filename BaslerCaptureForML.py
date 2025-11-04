import os
import sys
import cv2
import time
from vision_pipeline import Camera

os.environ["PYLON_CAMEMU"] = "3"

# Configuration
CONFIG_NAME = "correct_assembly_TEST"
NUM_ITERATIONS = 2
SAVE_DIR = "image_data/train"


def capture_training_photos(camera, config_name, iteration, save_dir):
    """
    Captures 2 photos automatically with 2s delay
    Press SPACE to start capture, ESC to cancel
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Iteration {iteration + 1} ===")
    print("Press SPACE to capture 2 photos, ESC to cancel")

    # Create window ONCE before loop
    cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)

    waiting_for_trigger = True
    photos_captured = 0

    while True:
        # Capture raw image
        image = camera.capture_raw()

        if image is None:
            print("ERROR: Failed to capture image")
            cv2.destroyAllWindows()
            return False

        # Display with instructions
        display_img = image.copy()

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
                filename = f"{save_dir}/{config_name}_{iteration:02d}_01.jpg"
                cv2.imwrite(filename, image)
                print(f"✓ Photo 1/2 saved: {filename}")
                photos_captured += 1

                # Wait 2 seconds before next capture
                time.sleep(2)

            elif photos_captured == 1:
                filename = f"{save_dir}/{config_name}_{iteration:02d}_02.jpg"
                cv2.imwrite(filename, image)
                print(f"✓ Photo 2/2 saved: {filename}")
                photos_captured += 1

                # Done with this iteration
                cv2.destroyAllWindows()
                return True


def main():
    """Main function to run capture session"""
    print(f"Configuration: {CONFIG_NAME}")
    print(f"Target iterations: {NUM_ITERATIONS}")
    print(f"Save location: {SAVE_DIR}")
    input("\nPress Enter to start capture session...")

    try:
        # Initialize camera once (with continuous grabbing)
        camera = Camera(exposure_time=30000.0, frame_rate=30.0)

        # Capture loop
        for iteration in range(NUM_ITERATIONS):
            print(f"\n{'=' * 50}")
            print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
            print(f"{'=' * 50}")
            input("Place assembly, press Enter...")

            success = capture_training_photos(camera, CONFIG_NAME, iteration, SAVE_DIR)

            if not success:
                print("Session cancelled by user")
                break

        print(f"\n✓ Capture session complete!")
        total_images = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
        print(f"Total images saved: {total_images}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        camera.shutdown()


if __name__ == "__main__":
    main()
