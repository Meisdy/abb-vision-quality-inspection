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
SHOW_IMAGES = False
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
    # Capture image from camera
    img_bgr = Camera.capture_raw()  # BGR, numpy array
    if img_bgr is None:
        logging.error("Failed to capture image")
        return False

    # Convert BGR -> RGB so preprocessing matches Image.open(...).convert('RGB')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.asarray(img_rgb, dtype=np.uint8)  # ensure dtype    if img is None:

    if SHOW_IMAGES:
        cv2.imshow("Captured Image", vision_pipeline.VisionProcessor.crop(img_bgr))
        cv2.waitKey(1)
        logging.info(f"Loaded model with classes: {ml_evaluator.classes}")

    # Classify using the trained model (expects RGB numpy)
    results, status = ml_evaluator.predict_one(img_rgb)
    status_str = "GOOD" if status else "BAD"
    logging.info(f"Prediction: {status_str}")
    for region, label, conf in results:
        # show raw confidence regardless of override
        logging.info(f"  {region}: {label:<20} conf: {conf * 100:.2f}%")

    # Return status
    return False if status == 'BAD' else True


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
