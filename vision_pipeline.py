"""
vision_pipeline.py

Provides classes and functions for camera handling (Basler/Pylon) and image processing
for industrial vision pipelines. Supports ROI selection, image loading, cropping, preprocessing,
and multiple augmentation routines, enabling flexible integration with vision-centric workflows.
"""

import os
import cv2
import numpy as np
from pypylon import pylon, genicam
import logging

# Configuration
ROI = (528, 63, 1221, 1096)  # ROI for border-cropped part images (x, y, w, h)


def setup_converter():
    """Setup image format converter for BGR."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


class Camera:
    """Handle all Basler camera and image preprocessing for the project."""

    def __init__(self, exposure_time: float = 30000.0, frame_rate: float = 30.0, max_cameras: int = 1):
        self.exposure_time = exposure_time
        self.frame_rate = frame_rate
        self.max_cameras = max_cameras
        self.converter = setup_converter()
        self.cameras = self.setup_cameras()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        logging.info("Camera initialized and grabbing started")

    def setup_cameras(self):
        """Initialize Basler camera."""
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        cameras = pylon.InstantCameraArray(min(len(devices), self.max_cameras))
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            logging.info(f"Using camera: {cam.GetDeviceInfo().GetModelName()}")
            if cam.GetDeviceInfo().GetModelName() == 'Emulation':
                logging.warning('No physical Camera connected!')

            cam.Open()
            cam.ExposureTime.SetValue(self.exposure_time)
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(self.frame_rate)
            cam.Close()
        return cameras

    def capture_raw(self):
        """Grab single raw image from camera."""
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

    def shutdown(self):
        """Cleanup camera."""
        if self.cameras:
            self.cameras.StopGrabbing()
        logging.info("Camera shutdown complete")


class VisionProcessor:
    """Static methods for loading, preprocessing, cropping, and augmenting images."""

    @staticmethod
    def load_images(folder: str):
        """Load all image files from folder.

        Args:
            folder (str): Path to images.

        Returns:
            tuple[np.ndarray, list[str]]: Tuple of images and corresponding filenames.
        """
        images = []
        filenames = []
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img)
                    filenames.append(filename)
        return np.array(images), filenames

    @staticmethod
    def load_images_npy(folder: str):
        """Load only .npy files from processed subfolder.

        Args:
            folder (str): Path to .npy files.

        Returns:
            np.ndarray: Array of .npy images.
        """
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

    @staticmethod
    def crop(image: np.ndarray, roi: tuple = ROI) -> np.ndarray:
        """Crop the image to roi=(x, y, w, h) and return result."""
        if image is None:
            raise ValueError("Input image is None")
        x, y, w, h = roi
        return image[y:y + h, x:x + w]

    @staticmethod
    def preprocess(image: np.ndarray, roi: tuple = ROI, resize_to: int = None, normalize: bool = False,
                   visualisation: bool = False) -> np.ndarray:
        """Crop, resize, and preprocess image for ML pipelines.

        Args:
            image (np.ndarray): Input image.
            roi (tuple, optional): ROI for cropping.
            resize_to (int, optional): Output resolution for resizing.
            normalize (bool, optional): If True, normalize to 0-1 and transpose to CHW.
            visualisation (bool, optional): If True, show the preprocessed image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        if roi:
            img = VisionProcessor.crop(image, roi)
        if resize_to:
            img = cv2.resize(img, (resize_to, resize_to))
        if visualisation:
            cv2.namedWindow('Preprocessed Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Preprocessed Image', img)
            cv2.resizeWindow('Preprocessed Image', 1000, 1000)
            cv2.waitKey(0)
            cv2.destroyWindow('Preprocessed Image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize:
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        return img

    @staticmethod
    def augment_for_anomaly(img: np.ndarray) -> np.ndarray:
        """Apply augmentations suitable for anomaly detection training.

        img should be uint8 HWC format (before normalization).
        """
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

    @staticmethod
    def augment_lighting(img: np.ndarray) -> np.ndarray:
        """Randomize brightness and contrast for lighting augmentation."""
        brightness_factor = np.random.uniform(0.9, 1.2)
        img_aug = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        contrast_factor = np.random.uniform(0.9, 1.1)
        mean = np.mean(img_aug, axis=(0, 1), keepdims=True)
        img_aug = np.clip((img_aug - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        return img_aug

    @staticmethod
    def augment_color_saturation(img: np.ndarray) -> np.ndarray:
        """Randomize hue, saturation, and value for color augmentation."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        hue_shift = np.random.randint(-8, 9)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        val_scale = np.random.uniform(0.95, 1.05)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
        img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_aug
