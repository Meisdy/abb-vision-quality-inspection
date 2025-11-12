import os
import cv2
import numpy as np
import logging
import abb_robot_comm
import vision_pipeline
import ML.classifier_evaluation as ml

os.environ["PYLON_CAMEMU"] = "3"

# Constants
USE_CAMERA = True
USE_ROBOT = True
IP_ABB_ROBOT = '192.168.125.5'
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


def evaluate_part(Camera, ml_evaluator) -> bool:
    logging.info("Starting part evaluation")

    # Capture image from camera
    img = Camera.capture_raw()  # BGR, numpy array

    if img is None:
        logging.error("Failed to capture image")
        return False

    # Convert BGR -> RGB so preprocessing matches Image.open(...).convert('RGB')
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.asarray(rgb_img, dtype=np.uint8)  # ensure dtype    if img is None:

    # Classify using the trained model
    logging.info(f"Loaded model with classes: {ml_evaluator.classes}")
    label, confidence = ml_evaluator.predict_one(rgb_img)  # pass raw image, gets processed in method
    logging.info(f"Classification result: {label.upper()} with confidence {confidence:.4f}")

    # Determine if part is correct or incorrect
    if confidence < 0.6:
        logging.warning("Low confidence in prediction, marking as incorrect mix")
        label = 'incorrect_mix'

    # Return status
    return False if label == 'incorrect_mix' else True


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
                    Robot.send_message("COMPLETE")
                else:
                    Robot.send_message("BAD_PART")

            logging.info('Client disconnected')

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        Camera.shutdown()
        logging.info("Program ended")


if __name__ == '__main__':
    main()
