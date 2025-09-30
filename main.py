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
IMAGE_NAME = 'Image6.jpg'
IP_ABB_ROBOT = '192.168.125.202'
Picture = []

# Camera Const
maxCamerasToUse = 1
exitCode = 0


def main():
    if USE_ROBOT:

        print("LOG: Using Robot. Setting up TCP Server now")
        Robot = abb_robot_comm.RobotComm(IP_ABB_ROBOT)
        Robot.connect()
        Robot.communicate("Hello from Python")
        Robot.disconnect()

        print("LOG: Using Robot. Setting up TCP Server now")

    else:
        print("LOG: Not using Robot. Skipping TCP Server setup")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
