import cv2
import time
import numpy as np

from tensorflow import keras

from Util import HandTrackingModule
from Util import BasicToolModule
from Util import ImageProcessingModule

# Start of Setting
##################
wCam, hCam = 480, 360  # width and height image
noCam = 0  # default Cam
globalColor = (255, 0, 0)  # default color
detectionCon = 0.70  # set Confident in AI Mediapipe

threshold = 0.70

cameraBrightness = 190  # Set Brightness
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

model = keras.models.load_model(basicTools.getBaseUrl() + '/Resources/model/model_trained.h5')

while True:
    # read image from cam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip the image

    # detect hand
    img, imgCanvas = detector.findHands(img)
    lmList, bboxList, bboxAll = detector.findPosition(img, draw=True)

    # Get ROI
    imgRoi = basicTools.CreateBlankImage(img)
    if len(bboxAll) > 0:
        # crop image in matrix y,x
        imgRoi = imgCanvas[abs(bboxAll[1] - 20): abs(bboxAll[3] + 20),
                 abs(bboxAll[0] - 20):abs(bboxAll[2] + 20)]
        imgRoiCopy = imgRoi.copy()

        # setting region of interest
        imgRoi = cv2.resize(imgRoi, (32, 32))
        imgRoi = imageProcessing.preProcessing(imgRoi)
        imgRoi = imgRoi.reshape(1, 32, 32, 1)

        # Predict
        classIndex = int(model.predict_classes(imgRoi))
        predictions = model.predict(imgRoi)
        proVal = np.amax(predictions)
        print(classIndex, proVal)

        # show Prediction
        if proVal >= threshold:
            cv2.putText(img, str(classIndex) + ', ' + str(proVal), (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1,
                        globalColor, 1)

    imgRoiCopy = cv2.resize(imgRoiCopy, (wCam, hCam))  # resize img region of interest

    # show text
    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([img, imgCanvas], [imgRoiCopy, imgRoi]))
    cv2.imshow("Stacked Image", stackedImages)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
