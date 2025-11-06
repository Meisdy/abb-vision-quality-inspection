import os
import cv2
import numpy as np
from pypylon import pylon, genicam
import logging

# Configuration
IMAGE_SIZE = 1024
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


def augment_for_anomaly(img):
    """Apply augmentations suitable for anomaly detection training.
    img should be uint8 HWC format (before normalization)"""

    # Random small rotation (±15 degrees)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random brightness adjustment
    if np.random.rand() < 0.6:
        brightness_factor = np.random.uniform(0.85, 1.15)
        img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)

    # Random contrast adjustment
    if np.random.rand() < 0.4:
        contrast_factor = np.random.uniform(0.9, 1.1)
        img = np.clip((img - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)

    return img


def load_images_npy(folder):
    """Load only .npy files from processed subfolder"""
    images = []

    if not os.path.exists(folder):
        print(f"Warning: {folder} not found")
        return np.array(images)

    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            filepath = os.path.join(folder, file)
            img = np.load(filepath)
            images.append(img)

    print(f"Found {len(images)} .npy files in {folder}")
    return np.array(images)


def load_images(folder):
    """Load all image files from folder."""
    images = []
    filenames = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)

    return np.array(images), filenames





