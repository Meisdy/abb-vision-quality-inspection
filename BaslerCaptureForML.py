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


def grab_training_images(cameras, converter, config_name, num_iterations, save_dir):
    """
    Captures training images with guided prompts
    3 photos per iteration: standard, standard, varied lighting
    """
    os.makedirs(save_dir, exist_ok=True)

    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    iteration = 0

    print(f"\n=== Starting capture session for '{config_name}' ===")
    print(f"Target: {num_iterations} iterations × 3 photos = {num_iterations * 3} images")
    print("\nControls:")
    print("  SPACE = Capture photo")
    print("  ESC = Exit\n")

    while cameras.IsGrabbing() and iteration < num_iterations:
        grabResult = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult).GetArray()

            # Display with instructions
            display_img = image.copy()
            text = f"Iteration {iteration + 1}/{num_iterations}"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera Feed', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to exit
                print("\nSession cancelled.")
                grabResult.Release()
                break

            elif key == 32:  # SPACE to capture
                # Determine which photo in the sequence (0, 1, or 2)
                photo_in_iteration = 0
                if os.path.exists(f"{save_dir}/{config_name}_{iteration:02d}_std1.jpg"):
                    photo_in_iteration = 1
                if os.path.exists(f"{save_dir}/{config_name}_{iteration:02d}_std2.jpg"):
                    photo_in_iteration = 2

                # Save with appropriate name
                if photo_in_iteration == 0:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_std1.jpg"
                    print(f"  ✓ Photo 1/3 saved (standard lighting)")
                elif photo_in_iteration == 1:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_std2.jpg"
                    print(f"  ✓ Photo 2/3 saved (standard lighting)")
                elif photo_in_iteration == 2:
                    filename = f"{save_dir}/{config_name}_{iteration:02d}_varied.jpg"
                    print(f"  ✓ Photo 3/3 saved (varied lighting)")
                    iteration += 1
                    if iteration < num_iterations:
                        print(f"\n--- Iteration {iteration + 1} ready ---")
                        print("  Place new assembly or adjust components")

                cv2.imwrite(filename, image)

        grabResult.Release()

    cameras.StopGrabbing()
    cv2.destroyAllWindows()

    total_captured = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    print(f"\n✓ Session complete! {total_captured} images saved to {save_dir}")


def main():
    try:
        # Configuration
        CONFIG_NAME = "correct_assembly"  # Change this for different configs
        NUM_ITERATIONS = 3  # Number of assemblies to photograph
        SAVE_DIR = "image_data/train"  # Where to save images

        print(f"Configuration: {CONFIG_NAME}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Save location: {SAVE_DIR}")
        input("\nPress Enter to start...")

        converter = configure_converter()
        cameras = create_cameras(exposure_time=30000.0, frame_rate=30.0)
        grab_training_images(cameras, converter, CONFIG_NAME, NUM_ITERATIONS, SAVE_DIR)

    except genicam.GenericException as e:
        print("Camera exception:", e)
        sys.exit(1)
    except OSError as e:
        print("File system error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
