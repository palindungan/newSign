import sys


class BasicTools():
    def __init__(self):
        self.pTime = 0

    def countFps(self, time):
        cTime = time
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        return fps

    def getBaseUrl(self):
        return sys.path[1]

    def empty(self, a):
        pass
