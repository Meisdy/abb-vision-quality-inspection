from pypylon import genicam
from pypylon import pylon
import os
import sys
import cv2
import time

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
    Captures 2 photos automatically with 2s delay
    Press SPACE to start capture, ESC to cancel
    """
    os.makedirs(save_dir, exist_ok=True)

    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    print(f"=== Iteration {iteration + 1} ===")
    print("Press SPACE to capture 2 photos, ESC to cancel")

    # Create window ONCE before loop
    cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)

    waiting_for_trigger = True
    photos_captured = 0

    while cameras.IsGrabbing():
        grabResult = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult).GetArray()

            # Display with instructions
            display_img = image.copy()

            if waiting_for_trigger:
                text = "Press SPACE to capture"
                cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                text = f"Capturing... Photo {photos_captured + 1}/2"
                cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Camera Feed', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Cancelled by user")
                grabResult.Release()
                cameras.StopGrabbing()
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
                    print(f"✓ Photo 1/2 saved")
                    photos_captured += 1
                    # Wait 2 seconds before next capture
                    time.sleep(2)

                elif photos_captured == 1:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_02.jpg"
                    cv2.imwrite(filename, image)
                    print(f"✓ Photo 2/2 saved")
                    photos_captured += 1
                    # Done with this iteration
                    break

        grabResult.Release()

    cameras.StopGrabbing()
    cv2.destroyAllWindows()
    return True


def main():
    """Main function to run capture session"""

    # Configuration
    CONFIG_NAME = "correct_assembly_TEST"
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

            if iteration == 0:
                input("Place assembly, press Enter...")
            else:
                input("Replace assembly, press Enter...")

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
