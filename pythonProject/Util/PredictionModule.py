import numpy as np

from tensorflow import keras

from Util import BasicToolModule
from Util import TranslationModule


class Prediction():
    def __init__(self):
        self.basicTools = BasicToolModule.BasicTools()
        self.translation = TranslationModule.Translation()

        self.model_numeric = keras.models.load_model(
            self.basicTools.getBaseUrl() + '/Resources/model/model_trained_numeric.h5')
        self.model_alphabet = keras.models.load_model(
            self.basicTools.getBaseUrl() + '/Resources/model/model_trained_alphabet.h5')

    def predict(self, imgRoi):
        # Predict NUMERIC
        numeric_classIndex = int(self.model_numeric.predict_classes(imgRoi))
        numeric_predictions = self.model_numeric.predict(imgRoi)
        numeric_proVal = np.amax(numeric_predictions)

        # Predict ALPHABET
        alphabet_classIndex = int(self.model_alphabet.predict_classes(imgRoi))
        alphabet_predictions = self.model_alphabet.predict(imgRoi)
        alphabet_proVal = np.amax(alphabet_predictions)

        if numeric_proVal > alphabet_proVal:
            classIndex = numeric_classIndex
            predictions = numeric_predictions
            proVal = numeric_proVal
            predictionType = 'NUMERIC'
        else:
            classIndex = alphabet_classIndex
            predictions = alphabet_predictions
            proVal = alphabet_proVal
            predictionType = 'ALPHABET'

        return classIndex, predictions, proVal, predictionType
