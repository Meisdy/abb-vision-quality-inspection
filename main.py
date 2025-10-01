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
USE_ROBOT = False
#IMAGE_NAME = 'Image6.jpg'
IP_ABB_ROBOT = '192.168.125.202'
Picture = []

# Camera Const
maxCamerasToUse = 1
exitCode = 0


def main():
    if USE_ROBOT:

        # Testing TCP Server with ABB Robot
        Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
        Robot.connect()
        answer = Robot.communicate("Vision System Ready")
        if answer == "Robot ready":
            print("LOG: Robot is ready. Proceeding with vision system tasks.")
        else:
            print("LOG: Robot did not respond as expected. Check connection.")
            sys.exit(1)

        # Wait for user input to disconnect
        user_input = input("Press Enter to disconnect from the robot...")
        Robot.disconnect()

    else:
        sys.exit(1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
