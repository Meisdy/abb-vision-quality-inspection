from pypylon import genicam
from pypylon import pylon
import os
import sys
import cv2

os.environ["PYLON_CAMEMU"] = "3"

maxCamerasToUse = 1


def configure_converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def create_cameras(exposure_time, frame_rate):
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Using device ", cam.GetDeviceInfo().GetModelName())
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)
        cam.Close()
    return cameras


def capture_training_photos(cameras, converter, config_name, iteration, save_dir):
    """
    Captures 3 photos (std, std, varied) with user pressing SPACE
    Returns True when all 3 photos captured, False if cancelled
    """
    os.makedirs(save_dir, exist_ok=True)

    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    photos_captured = 0

    print(f"=== Iteration {iteration + 1} - Capture {photos_captured + 1}/3 ===")
    print("Press SPACE to capture, ESC to cancel")

    while cameras.IsGrabbing() and photos_captured < 3:
        grabResult = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult).GetArray()

            # Display with instructions
            display_img = image.copy()
            text = f"Photo {photos_captured + 1}/3"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera Feed', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Cancelled by user")
                grabResult.Release()
                cameras.StopGrabbing()
                cv2.destroyAllWindows()
                return False

            elif key == 32:  # SPACE
                # Save photo based on sequence
                if photos_captured == 0:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_std1.jpg"
                    print("✓ Photo 1/3 saved (standard)")
                elif photos_captured == 1:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_std2.jpg"
                    print("✓ Photo 2/3 saved (standard)")
                elif photos_captured == 2:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_varied.jpg"
                    print("✓ Photo 3/3 saved (varied)")

                cv2.imwrite(filename, image)
                photos_captured += 1

                if photos_captured < 3:
                    if photos_captured == 2:
                        print(">>> Adjust lighting for photo 3, then press SPACE <<<")

        grabResult.Release()

    cameras.StopGrabbing()
    cv2.destroyAllWindows()
    return True


def main():
    """Main function to run capture session"""

    # Configuration
    CONFIG_NAME = "correct_assembly"
    NUM_ITERATIONS = 2
    SAVE_DIR = "image_data/train"

    print(f"Configuration: {CONFIG_NAME}")
    print(f"Target iterations: {NUM_ITERATIONS}")
    print(f"Save location: {SAVE_DIR}")
    input("\nPress Enter to start capture session...")

    try:
        # Initialize camera
        converter = configure_converter()
        cameras = create_cameras(exposure_time=30000.0, frame_rate=30.0)

        # Capture loop
        for iteration in range(NUM_ITERATIONS):
            print(f"\n{'=' * 50}")
            print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
            print(f"{'=' * 50}")
            input("Place assembly manually, then press Enter...")

            success = capture_training_photos(
                cameras,
                converter,
                CONFIG_NAME,
                iteration,
                SAVE_DIR
            )

            if not success:
                print("Session cancelled by user")
                break

            input("\nRemove assembly, press Enter to continue...")

        print(f"\n✓ Capture session complete!")
        total_images = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
        print(f"Total images saved: {total_images}")

    except genicam.GenericException as e:
        print(f"Camera error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
