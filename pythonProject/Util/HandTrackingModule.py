import cv2
import mediapipe as mp
import time
import math
import numpy as np


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

    def findHands(self, img, draw=True):
        width = img.shape[1]
        height = img.shape[0]
        imgCanvas = np.zeros((width, height, 3), np.uint8)

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

        return img, imgCanvas

    def findPosition(self, img, handNo=0, draw=True):

        xList = []
        yList = []
        bbox = []

        # declaration
        self.lmList = []

        # detect if there is hand or not
        if self.results.multi_hand_landmarks:

            # draw multiple hand
            for myHand in self.results.multi_hand_landmarks:

                xList = []
                yList = []
                bbox = []

                # detect index ,position (ratio) landmark  in image
                for id, lm in enumerate(myHand.landmark):

                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # get all x, y position
                    xList.append(cx)
                    yList.append(cy)

                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    # if id == 0:
                    #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # finc min and max each x y
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)

                bbox = xMin, yMin, xMax, yMax

                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []

        # thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            # print('Index Finger Open')
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # print('Index Finger Open')
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
