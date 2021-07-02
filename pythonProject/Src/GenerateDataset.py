import cv2
import time
import numpy as np
import os
from Util import HandTrackingModule
from Util import BasicToolModule
from Util import ImageProcessingModule

# Start of Setting
##################
wCam, hCam = 480, 360  # width and height image
noCam = 0  # default Cam
globalColor = (255, 0, 0)  # default color
detectionCon = 0.70  # set Confident in AI Mediapipe

cameraBrightness = 190  # Set Brightness
moduleVal = 10  # SAVE EVERY 1 FRAME TO AVOID REPETITION
minBlur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False  # IMAGE SAVED COLORED OR GRAY
saveData = True  # SAVE DATA FLAG
showImage = True  # IMAGE DISPLAY FLAG
imgWidth = 180  # Resize width Image
imgHeight = 120  # Resize height Image
##################
# End of Setting

# Start of Declare Object Class
basicTools = BasicToolModule.BasicTools()
imageProcessing = ImageProcessingModule.ImageProcessing()
detector = HandTrackingModule.HandDetector(detectionCon=detectionCon, maxHands=2)
# End of Declare Object Class

# Start of Set
cap = cv2.VideoCapture(noCam)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10, cameraBrightness)

myPath = basicTools.getBaseUrl() + '/Resources/dataset/'  # PATH TO SAVE IMAGE

count = 0
countSave = 0
# End of Set

# # make folder
# if saveData:
#     basicTools.CreateDirectory(myPath)

while True:
    # read image from cam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip the image
    imgCopy = img.copy()

    # detect hand
    img = detector.findHands(img)
    lmList, bboxList, bboxAll = detector.findPosition(img, draw=True)

    # show fps
    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([img]))
    cv2.imshow("Stacked Image", stackedImages)

    if len(bboxAll) > 0:
        cv2.rectangle(imgCopy, (bboxAll[0] - 20, bboxAll[1] - 20), (bboxAll[2] + 20, bboxAll[3] + 20), (0, 255, 0),
                      2)
    cv2.imshow("imgCopy", imgCopy)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
