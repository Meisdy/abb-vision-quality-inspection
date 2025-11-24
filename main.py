"""
main.py

Controls the vision inspection process, connects to ABB robot, manages camera image capture,
classifies images using trained ML model, and communicates results via robot interface.
Includes logging setup and error handling.
"""

import os
import cv2
import time
import logging
import abb_robot_comm
import vision_pipeline
import ML.classifier_evaluation as ml

IP_ABB_ROBOT = '192.168.125.5'
DEFAULT_EXPOSURE = 30000.0
DEFAULT_FRAMERATE = 30.0
os.environ["PYLON_CAMEMU"] = "3"


class WhiteInfoFormatter(logging.Formatter):
    """Custom formatter for info-level white console output."""
    WHITE = '\033[97m'
    RESET = '\033[0m'

    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.INFO:
            return f"{self.WHITE}{msg}{self.RESET}"
        return msg


def setup_logging():
    """Configure logging with custom formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(WhiteInfoFormatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)


def evaluate_part(Camera, ml_evaluator):
    """
    Capture and classify an image from Camera using the ML model function predict_one.
    Returns: (status: bool, img_bgr: np.ndarray)
    """
    img_bgr = Camera.capture_raw()  # BGR, numpy array
    if img_bgr is None:
        logging.error("Failed to capture image")
        return False, None

    # Classify using the trained model (expects RGB numpy, so set input_is_bgr=True)
    results, status = ml_evaluator.predict_one(img_bgr, input_is_bgr=True)
    result_items = [f"{region}: {label:<15} conf: {conf * 100:.2f}%" for region, label, conf in results]
    logging.info(f"Prediction: {'GOOD' if status else 'BAD'}: {result_items}")
    return status, img_bgr


def main():
    setup_logging()
    try:
        Camera = vision_pipeline.Camera(exposure_time=DEFAULT_EXPOSURE, frame_rate=DEFAULT_FRAMERATE)
        ml_evaluator = ml.ClassifierEvaluator()
        logging.info(f"Loaded model with classes: {ml_evaluator.classes}")
        logging.info(f"Model info: {ml_evaluator.model_info}")

        while True:
            Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
            Robot.connect()
            Robot.communicate("Vision System Ready")
            if Robot.receive_message() == "evaluate":
                logging.info("Starting part evaluation for client")
                part_status, image = evaluate_part(Camera, ml_evaluator)
                Robot.send_message("COMPLETE" if part_status else "BAD_PART")
                logging.info('Client disconnected')  # Only runs after client quits
                cv2.imshow("Captured Image", vision_pipeline.VisionProcessor.crop(image))
                time.sleep(10)
                cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        Camera.shutdown()
        logging.info("Program ended")


if __name__ == '__main__':
    main()
