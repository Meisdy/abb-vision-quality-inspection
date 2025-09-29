import os
from pypylon import genicam
from pypylon import pylon
import sys
import cv2

os.environ["PYLON_CAMEMU"] = "3"

'''
Program to grab images upon a keystroke from both cameras and save them
'''
maxCamerasToUse = 1

exitCode = 0


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
        # Set the exposure time
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)
        cam.Close()
    return cameras


def grab_images(cameras, converter):
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    frame_counter = 0
    image_counter = 0

    while cameras.IsGrabbing():
        grabResult1 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # grabResult2 = cameras[1].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult1.GrabSucceeded():
            frame_counter += 1
            image1 = converter.Convert(grabResult1).GetArray()

            cv2.namedWindow('Camera 1 Feed', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('Camera 2 Feed', cv2.WINDOW_NORMAL)

            # Show each camera feed in its respective window
            cv2.imshow('Camera 1 Feed', image1)
            # cv2.imshow('Camera 2 Feed', image2)

            if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
                break
            elif cv2.waitKey(1) == 32:  # Press 'Space' to save the image
                image_counter += 1
                filename = f"camera1_image_{image_counter}.png"
                cv2.imwrite(filename, image1)
                print(f"Image saved as {filename}")

    cameras.StopGrabbing()
    cv2.destroyAllWindows()


def main():
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
