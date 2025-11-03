import logging
import cv2
import os
import abb_robot_comm
import time
from pypylon import genicam
from pypylon import pylon
from inference import evaluate_image, get_status

os.environ["PYLON_CAMEMU"] = "3"

# Constants
USE_CAMERA = True
USE_ROBOT = True
IP_ABB_ROBOT = '192.168.125.5'
maxCamerasToUse = 1

# Global camera variables
cameras = None
converter = None


def setup_logging():
    class WhiteInfoFormatter(logging.Formatter):
        WHITE = '\033[97m'
        RESET = '\033[0m'

        def format(self, record):
            msg = super().format(record)
            if record.levelno == logging.INFO:
                return f"{self.WHITE}{msg}{self.RESET}"
            return msg

    handler = logging.StreamHandler()
    handler.setFormatter(WhiteInfoFormatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)


def configure_converter():
    """Setup image format converter for BGR"""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def setup_cameras(exposure_time=30000.0, frame_rate=30.0):
    """Initialize Basler camera"""
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        logging.info(f"Using camera: {cam.GetDeviceInfo().GetModelName()}")

        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)
        cam.Close()

    return cameras


def capture_image():
    """Capture single image from camera (already grabbing)"""
    global cameras, converter

    try:
        grabResult = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult).GetArray()
            grabResult.Release()
            return image
        else:
            logging.error("Failed to grab image")
            grabResult.Release()
            return None

    except genicam.GenericException as e:
        logging.error(f"Camera error: {e}")
        return None


def evaluate_part():
    """Capture image and evaluate for defects"""
    logging.info("Evaluating part...")

    # Capture image from camera
    time.sleep(2)

    img = capture_image()
    cv2.imshow('Image Captured',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    if img is None:
        logging.error("Failed to capture image")
        return False

    # Evaluate using trained model
    loss = evaluate_image(img)
    status = get_status(loss)

    logging.info(f"Loss: {loss:.6f} → {status}")

    return status == "OK"


def main():
    global cameras, converter

    setup_logging()

    try:
        # Initialize camera ONCE
        converter = configure_converter()
        cameras = setup_cameras(exposure_time=30000.0, frame_rate=30.0)
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # Start once here

        logging.info("Camera initialized")

        while True:
            Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
            Robot.connect()
            Robot.communicate("Vision System Ready")

            if Robot.receive_message() == "evaluate":
                logging.info("Received evaluate command from robot.")
                part_status = evaluate_part()
                if part_status:
                    Robot.send_message("complete")
                else:
                    Robot.send_message("part bad")

            time.sleep(2)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if cameras:
            cameras.StopGrabbing()  # Stop once at end
        logging.info("Program ended")


if __name__ == '__main__':
    main()