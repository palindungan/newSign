import cv2
import time

from Util import HandTrackingModule
from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import TranslationModule
from Util import PredictionModule

# Start of Setting
##################
wCam, hCam = 480, 360  # width and height image
noCam = 0  # default Cam
globalColor = (255, 0, 0)  # default color
detectionCon = 0.80  # set Confident in AI Mediapipe

threshold = 0.9

cameraBrightness = 190  # Set Brightness

imageDimensions = (32, 32, 3)
##################
# End of Setting

# Start of Declare Object Class
basicTools = BasicToolModule.BasicTools()
imageProcessing = ImageProcessingModule.ImageProcessing()
detector = HandTrackingModule.HandDetector(detectionCon=detectionCon, maxHands=2)
translation = TranslationModule.Translation()
prediction = PredictionModule.Prediction()
# End of Declare Object Class

# Start of Set
cap = cv2.VideoCapture(noCam)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10, cameraBrightness)

while True:
    # read image from cam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip the image

    # detect hand
    img, imgCanvas = detector.findHands(img)
    lmList, bboxList, bboxAll = detector.findPosition(img, draw=True)

    # Get ROI
    imgRoi = basicTools.CreateBlankImage(img)
    imgRoiCopy = basicTools.CreateBlankImage(img)
    imgRoiCNN = basicTools.CreateBlankImage(img)
    if len(bboxAll) > 0:
        # crop image in matrix y,x
        imgRoi = imgCanvas[abs(bboxAll[1] - 20): abs(bboxAll[3] + 20),
                 abs(bboxAll[0] - 20):abs(bboxAll[2] + 20)]
        imgRoiCopy = imgRoi.copy()

        # setting region of interest
        imgRoi = cv2.resize(imgRoi, (imageDimensions[0], imageDimensions[1]))
        imgRoi = imageProcessing.preProcessing(imgRoi)
        imgRoiCNN = cv2.resize(imgRoi, (wCam, hCam))
        imgRoi = imgRoi.reshape(1, imageDimensions[0], imageDimensions[1], 1)

        # Predict
        classIndex, predictions, proVal, predictionType = prediction.predictAll(imgRoi)

        # show Prediction
        if proVal >= threshold:
            cv2.putText(img, translation.mapper(classIndex, predictionType) + ' (' + str(proVal) + ')', (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        globalColor, 3)

    imgRoiCopy = cv2.resize(imgRoiCopy, (wCam, hCam))  # resize img region of interest

    # show text
    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([img, imgCanvas], [imgRoiCopy, imgRoiCopy]))
    cv2.imshow("Stacked Image", stackedImages)

    cv2.imshow('imgRoiCNN', imgRoiCNN)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
