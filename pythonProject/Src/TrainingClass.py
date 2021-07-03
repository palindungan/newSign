import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D

from Util import HandTrackingModule
from Util import BasicToolModule
from Util import ImageProcessingModule


class TrainingClass():

    def getDatasetArray(self, path, imageDimensions):
        images = []  # contain all images
        classNo = []  # class of images

        myList = os.listdir(path)  # get list content of folder => 0,1,2,3,4,5,6,7,8,9
        noOfClasses = len(myList)  # get number of folder => 10
        print('Total No of Classes Detected = ', noOfClasses)
        print('1 Importing Classes ........')

        # for in each folder 0,1,2,3,4,5,6,7,8,9
        for x in range(0, noOfClasses):
            myPicList = os.listdir(path + '/' + str(x))  # get all image in class folder
            # print(myPicList)
            print(x, end=' ')

            # for in each file on folder img001-00001 ... img001-01016 --> img010-01016
            for y in myPicList:
                curImg = cv2.imread(path + '/' + str(x) + '/' + y)  # read each file image
                curImg = cv2.resize(curImg,
                                    (imageDimensions[0],
                                     imageDimensions[1]))  # resize image to decrease computation cost
                images.append(curImg)  # add image matrix in list
                classNo.append(x)  # add class in List
                # print(images)

        print(' ')
        print('Number of Images = ' + str(len(images)))
        print('Number of Classes = ' + str(len(classNo)))

        # convert list to array
        images = np.array(images)
        classNo = np.array(classNo)

        print('shape = ' + str(images.shape))  # (10160, 32, 32, 3)
        print('shape = ' + str(classNo.shape))  # (10160,)

        return images, classNo

    def splitDataset(self, images, classNo, testRatio, valRatio):
        print(' ')
        print('1 Split Training dan Testing ........')
        # splitting data training and testing
        X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)

        print('Test Ratio = ' + str(testRatio))  # ration

        print('X Train = ' + str(X_train.shape))  # images
        print('Y Train = ' + str(y_train.shape))  # classes

        print('X Test = ' + str(X_test.shape))  # images
        print('Y Test = ' + str(y_test.shape))  # classes

        print('2 Split Training dan Validation ........')
        # splitting data training and validation
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

        print('X Training = ' + str(X_train.shape))  # images
        print('Y Training = ' + str(y_train.shape))  # classes

        print('X Validation = ' + str(X_validation.shape))  # images
        print('Y Validation = ' + str(y_validation.shape))  # classes

        print('3 Final Result of Spliting Data : ........')

        print('X Training = ' + str(X_train.shape))  # images
        print('Y Training = ' + str(y_train.shape))  # classes

        print('X Testing = ' + str(X_test.shape))  # images
        print('Y Testing = ' + str(y_test.shape))  # classes

        print('X Validation = ' + str(X_validation.shape))  # images
        print('Y Validation = ' + str(y_validation.shape))  # classes

        return X_train, y_train, X_test, y_test, X_validation, y_validation


def main():
    ##########################
    # Start of SETTING
    pathLabels = ''
    testRatio = 0.2
    valRatio = 0.2
    imageDimensions = (32, 32, 3)
    batchSizeVal = 50
    epochsVal = 1
    stepsPerEpochVal = 2000
    # End of SETTING
    ##########################

    # Start of Declare Object Class
    trainingClass = TrainingClass()
    basicTools = BasicToolModule.BasicTools()
    imageProcessing = ImageProcessingModule.ImageProcessing()
    # End of Declare Object Class

    # Start of Set
    path = basicTools.getBaseUrl() + '/Resources/dataset/'  # PATH IMAGE
    # End of Set

    ##########################
    # import dataset 0-9, create one row images array and labels array
    # start
    ##########################
    images, classNo = trainingClass.getDatasetArray(path, imageDimensions)

    ##########################
    # splitting and shuffle the data
    # start
    ##########################
    X_train, y_train, X_test, y_test, X_validation, y_validation = trainingClass.splitDataset(images, classNo,
                                                                                              testRatio, valRatio)


if __name__ == "__main__":
    main()
