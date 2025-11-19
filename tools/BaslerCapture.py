"""
BaslerCapture.py

Interactive tool for capturing images from Basler cameras using the pypylon SDK.
Images are displayed in real-time, and can be saved with a key press.
Intended for manual image acquisition and testing.
"""

import os
import sys
import cv2
from pypylon import genicam, pylon

os.environ["PYLON_CAMEMU"] = "3"
MAX_CAMERAS_TO_USE = 1


def configure_converter():
    """Configure pypylon converter for BGR output."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def create_cameras(exposure_time: float, frame_rate: float):
    """
    Set up connected Basler cameras and configure acquisition.

    Args:
        exposure_time (float): Exposure time to use.
        frame_rate (float): Acquisition frame rate.
    Returns:
        pylon.InstantCameraArray: Array of initialized cameras.
    """
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")
    cameras = pylon.InstantCameraArray(min(len(devices), MAX_CAMERAS_TO_USE))
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Using device", cam.GetDeviceInfo().GetModelName())
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)
        cam.Close()
    return cameras


def grab_images(cameras, converter):
    """
    Show realtime camera feed; save image on space, exit on ESC.
    Images are saved as PNGs in the current directory.
    """
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    frame_counter = 0
    image_counter = 0
    while cameras.IsGrabbing():
        grabResult1 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult1.GrabSucceeded():
            frame_counter += 1
            image1 = converter.Convert(grabResult1).GetArray()
            cv2.namedWindow('Camera 1 Feed', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera 1 Feed', image1)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                grabResult1.Release()
                break
            elif key == 32:  # Space to save the image
                image_counter += 1
                filename = f"camera1_image_{image_counter}.png"
                cv2.imwrite(filename, image1)
                print(f"Image saved as {filename}")
            grabResult1.Release()
    cameras.StopGrabbing()
    cv2.destroyAllWindows()


def main():
    """
    Entry point for camera capture script. Handles exceptions.
    """
    try:
        converter = configure_converter()
        cameras = create_cameras(exposure_time=30000.0, frame_rate=30.0)
        grab_images(cameras, converter)
    except genicam.GenericException as e:
        print("An exception occurred.", e)
        sys.exit(1)
    except OSError as e:
        print("Disk space error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
