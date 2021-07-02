import cv2
import os
import time
import numpy as np
import math
from Util import HandTrackingModule
from Util import BasicToolModule

# Start of Setting
##################
wCam, hCam = 480, 360  # width and height image
noCam = 0
globalColor = (255, 0, 0)
##################
# End of Setting

cap = cv2.VideoCapture(noCam)
cap.set(3, wCam)
cap.set(4, hCam)

basicTools = BasicToolModule.BasicTools()
detector = HandTrackingModule.HandDetector(detectionCon=0.65, maxHands=2)

imgCanvas = np.zeros((480, 360, 3), np.uint8)

while True:
    success, img = cap.read()

    img, imgCanvas = detector.findHands(img)
    lmList, bboxList = detector.findPosition(img, draw=True)

    print(bboxList)

    fps = basicTools.countFps(time=time.time())

    cv2.putText(img, f'FPS {int(fps)}', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)
    stackedImages = basicTools.stackImages(1, ([img, imgCanvas]))
    cv2.imshow("Stacked Image", stackedImages)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
