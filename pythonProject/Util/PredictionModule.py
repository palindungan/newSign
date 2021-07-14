import numpy as np

from tensorflow import keras

from Util import BasicToolModule
from Util import TranslationModule


class Prediction():
    def __init__(self):
        self.basicTools = BasicToolModule.BasicTools()
        self.translation = TranslationModule.Translation()

        self.model = keras.models.load_model(self.basicTools.getBaseUrl() + '/Resources/model/model_trained_numeric.h5')

    def predict(self, imgRoi, threshold):
        # Predict
        classIndex = int(self.model.predict_classes(imgRoi))
        predictions = self.model.predict(imgRoi)
        proVal = np.amax(predictions)

        predictionType = 'NUMERIC'

        # show Prediction
        if proVal >= threshold:
            return classIndex, predictions, proVal, predictionType
