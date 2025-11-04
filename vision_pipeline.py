import os
import cv2
import numpy as np
from pypylon import pylon, genicam
import logging

# Configuration
IMAGE_SIZE = 512
CROP_X = 0.15
CROP_Y = 0.02
CROP_W = 0.6
CROP_H = 0.6


class Camera:
    """Handle all Basler camera and image preprocessing for the Project"""

    def __init__(self, exposure_time=30000.0, frame_rate=30.0, max_cameras=1):
        self.exposure_time = exposure_time
        self.frame_rate = frame_rate
        self.max_cameras = max_cameras

        # Setup camera
        self.converter = self.setup_converter()
        self.cameras = self.setup_cameras()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        logging.info("Camera initialized and grabbing started")

    def setup_converter(self):
        """Setup image format converter for BGR"""
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return converter

    def setup_cameras(self):
        """Initialize Basler camera"""
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        cameras = pylon.InstantCameraArray(min(len(devices), self.max_cameras))

        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            logging.info(f"Using camera: {cam.GetDeviceInfo().GetModelName()}")

            cam.Open()
            cam.ExposureTime.SetValue(self.exposure_time)
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(self.frame_rate)
            cam.Close()

        return cameras

    def capture_raw(self):
        """Grab single raw image from camera"""
        try:
            grabResult = self.cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult).GetArray()
                grabResult.Release()
                return image
            else:
                logging.error("Failed to grab image")
                grabResult.Release()
                return None

        except genicam.GenericException as e:
            logging.error(f"Camera error: {e}")
            return None

    @staticmethod
    def preprocess(img, visualisation=False):
        """Crop and normalize image for ML pipeline"""
        if img is None:
            logging.warning("Image is None, skipping preprocessing")
            return None

        height, width = img.shape[:2]
        crop_x = int(width * CROP_X)
        crop_y = int(height * CROP_Y)
        crop_w = int(width * CROP_W)
        crop_h = int(height * CROP_H)

        img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Show BEFORE normalization (if needed)
        if visualisation:
            cv2.namedWindow('Preprocessed Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Preprocessed Image', img)
            cv2.resizeWindow('Preprocessed Image', 800, 800)  # ✅ Bigger window
            cv2.waitKey(0)
            cv2.destroyWindow('Preprocessed Image')

        # THEN do normalization + transpose
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        return img

    def capture_and_preprocess(self):
        """Capture + preprocess in one call"""
        raw = self.capture_raw()
        return self.preprocess(raw)

    def shutdown(self):
        """Cleanup camera"""
        if self.cameras:
            self.cameras.StopGrabbing()
        logging.info("Camera shutdown complete")


def load_images(folder):
    """Load only .npy files (already preprocessed)"""
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            img = np.load(os.path.join(folder, file))
            images.append(img)
    return np.array(images)
