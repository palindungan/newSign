import cv2
import os
import time
import numpy as np
import math
from Util import HandTrackingModule
from Util import BasicToolModule

# Start of Setting
##################
wCam, hCam = 1080, 720  # width and height image
noCam = 0
globalColor = (255, 0, 0)
##################
# End of Setting

cap = cv2.VideoCapture(noCam)
cap.set(3, wCam)
cap.set(4, hCam)

basicTools = BasicToolModule.BasicTools()
detector = HandTrackingModule.HandDetector(detectionCon=0.7, maxHands=2)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    fps = basicTools.countFps(time=time.time())

    cv2.putText(img, f'FPS {int(fps)}', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)
    cv2.imshow('Original Image', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
