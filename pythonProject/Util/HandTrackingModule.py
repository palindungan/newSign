import cv2
import mediapipe as mp
import math
import numpy as np
from Util import BasicToolModule


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # declaration before using mediapipe
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,
                                        self.trackCon)  # module for hand tracking and detection
        self.mpDraw = mp.solutions.drawing_utils  # module for drawing landmark connection
        self.tipIds = [4, 8, 12, 16, 20]

        self.basicTools = BasicToolModule.BasicTools()

    def findHands(self, img, draw=True):
        # create black image
        width = img.shape[1]
        height = img.shape[0]
        imgCanvas = np.zeros((width, height, 3), np.uint8)
        handLmsList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB

        self.results = self.hands.process(imgRGB)  # preform the hand detection
        # print(results.multi_hand_landmarks)

        # detect if there is hand or not
        if self.results.multi_hand_landmarks:

            # detect multiple hands
            for handLms in self.results.multi_hand_landmarks:

                # check if want to draw
                if draw:
                    # drawing connection landmark
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(imgCanvas, handLms, self.mpHands.HAND_CONNECTIONS)
                    handLmsList.append(handLms)

        return img, imgCanvas, handLmsList

    def findPosition(self, img, handNo=0, draw=True):
        # declaration
        self.lmList = []
        bboxList = []

        # detect if there is hand or not
        if self.results.multi_hand_landmarks:

            # draw multiple hand
            for idMyHand, myHand in enumerate(self.results.multi_hand_landmarks):

                xList = []
                yList = []

                # detect left right hand
                leftRightClassification = self.detectLeftRightHand(idMyHand)

                # detect index ,position (ratio) landmark  in image
                for idLandmark, lm in enumerate(myHand.landmark):

                    # print(idLandmark, lm)
                    h, w, c = img.shape  # get image shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # get coordinate

                    # get all x, y position
                    xList.append(cx)
                    yList.append(cy)

                    # add lanmark list object
                    self.lmList.append([idMyHand, idLandmark, cx, cy, leftRightClassification])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                        # detect wrist
                        if idLandmark == 0:
                            cv2.putText(img, leftRightClassification, (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 255), 3)

                # find min and max each x y
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)

                bbox = xMin, yMin, xMax, yMax
                bboxList.append(bbox)

                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bboxList

    def findDistance(self, idMyHand, p1, p2, img, draw=True):
        x1, y1 = self.lmList[idMyHand][p1][1], self.lmList[idMyHand][p1][2]
        x2, y2 = self.lmList[idMyHand][p2][1], self.lmList[idMyHand][p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def detectLeftRightHand(self, idMyHand):
        output = "None"
        for idx, classification in enumerate(self.results.multi_handedness):
            if classification.classification[0].index == idMyHand:
                # process result
                index = classification.classification[0].index  # 0 = left, 1 right
                label = classification.classification[0].label  # Left, Right
                score = classification.classification[0].score  # confident
                text = '{} {}'.format(label, round(score, 2))

                output = label

        return output

    def drawHandLandmarks(self, img, handLms):
        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
