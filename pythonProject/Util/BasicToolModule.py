import sys
import os


class BasicTools():
    def __init__(self):
        self.pTime = 0
        self.countFolder = 0

    def countFps(self, time):
        cTime = time
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        return fps

    def getBaseUrl(self):
        return sys.path[1]

    def empty(self, a):
        pass

    def CreateDirectory(self, myPath):
        self.countFolder = 0

        while os.path.exists(myPath + str(self.countFolder)):
            self.countFolder = self.countFolder + 1

        os.makedirs(myPath + str(self.countFolder))
