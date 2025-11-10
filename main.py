import logging
import cv2
import os
import time
import abb_robot_comm
import vision_pipeline
from ML.inference import evaluate_image, get_status

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


def evaluate_part(Camera):
    """Capture image and evaluate for defects"""
    logging.info("Evaluating part...")

    # Capture image from camera
    img = Camera.capture_raw()
    img = Camera.preprocess(img, True)
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
    setup_logging()

    try:
        # Initialize camera once
        Camera = vision_pipeline.Camera(exposure_time=30000.0, frame_rate=30.0)

        while True:
            Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
            Robot.connect()
            Robot.communicate("Vision System Ready")

            if Robot.receive_message() == "evaluate":
                logging.info("Received evaluate command from robot.")
                part_status = evaluate_part(Camera)
                if part_status:
                    Robot.send_message("complete")
                else:
                    Robot.send_message("part bad")

            time.sleep(2)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        Camera.shutdown()
        logging.info("Program ended")


if __name__ == '__main__':
    main()
