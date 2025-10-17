import logging
import cv2
import socket
import numpy as np
import math
import sys
import os
import abb_robot_comm
from pypylon import genicam
from pypylon import pylon

os.environ["PYLON_CAMEMU"] = "3"

# Constants
USE_CAMERA = False
USE_ROBOT = True
IP_ABB_ROBOT = '192.168.125.6'

# Camera Const
maxCamerasToUse = 1
exitCode = 0


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


def evaluate_part():
    # Placeholder for part evaluation logic
    logging.info("Evaluating part...")
    return False


def main():
    # Setup Logging
    setup_logging()

    # Setup Robot Communication
    Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
    Robot.connect()
    Robot.communicate("Vision System Ready")

    # Wait for evaluate command
    if Robot.receive_message() == "evaluate":
        logging.info("Received evaluate command from robot.")
        part_status = evaluate_part()
        if part_status:
            Robot.send_message("complete")
        else:
            Robot.send_message("part bad")

  #  wait = input("Press Enter to disconnect and exit...")
  #  Robot.disconnect()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
