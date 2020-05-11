from abc import ABC, abstractmethod


class Predictor(ABC):

    def predict_individual(self, image):
        return self.predict_batch(image.unsqueeze(0)).squeeze(0)

    @abstractmethod
    def predict_batch(self, image_array):
        pass


class ModulePredictor(Predictor):

    def __init__(self, module):
        self._module = module

    def predict_batch(self, image_array):
        return self._module(image_array).argmax(dim=1)
