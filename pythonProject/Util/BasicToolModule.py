# Start of import library
import cv2
import numpy as np
import sys


# End of import library

class BasicTools():
    def __init__(self):
        self.pTime = 0

    def countFps(self, time):
        cTime = time
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        return fps
