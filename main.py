import os
import cv2
import torch
import logging
import abb_robot_comm
import vision_pipeline
import ML.classifier_evaluation as ml
from vision_pipeline import VisionProcessor

os.environ["PYLON_CAMEMU"] = "3"

# Constants
USE_CAMERA = True
USE_ROBOT = True
IP_ABB_ROBOT = '192.168.125.201'
maxCamerasToUse = 1


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


def evaluate_part(Camera, ml_evaluator) -> str:
    logging.info("Starting part evaluation")

    # Capture image from camera
    img = Camera.capture_raw()  # BGR, numpy array
    if img is None:
        logging.error("Failed to capture image")
        return False

    # Classify using the trained model
    logging.info(f"Loaded model with classes: {ml_evaluator.classes}")
    label, confidence = ml_evaluator.predict_one(img)  # pass raw image, gets processed in method
    logging.info(f"Classification result: {label} with confidence {confidence:.4f}")

    # Get all classes from the evaluator but incorrect_mix
    good_classes = [cls for cls in ml_evaluator.classes if cls != 'incorrect_mix']

    # Return status
    return 'complete' if label in good_classes else 'part bad'


def main():
    setup_logging()

    try:
        # Initialize camera once
        Camera = vision_pipeline.Camera(exposure_time=30000.0, frame_rate=30.0)
        ml_evaluator = ml.ClassifierEvaluator()

        while True:
            Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
            Robot.connect()
            Robot.communicate("Vision System Ready")

            if Robot.receive_message() == "evaluate":
                logging.info("Received evaluate command from robot.")
                part_status = evaluate_part(Camera, ml_evaluator)
                if part_status:
                    Robot.send_message("complete")
                else:
                    Robot.send_message("part bad")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        Camera.shutdown()
        logging.info("Program ended")


if __name__ == '__main__':
    main()
