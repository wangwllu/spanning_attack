import torch.nn as nn
from abc import ABC, abstractmethod


class LossEvaluator(ABC):

    @abstractmethod
    def compute_batch(self, model, image_array, label_array):
        pass

    def compute_individual(self, model, image, label):
        return self.compute_batch(
            model, image.unsqueeze(0), label.unsqueeze(0)
        ).squeeze(0)


class CrossEntropyLossEvaluator(LossEvaluator):
    def __init__(self):
        self._criterion = nn.CrossEntropyLoss(reduction='none')

    def compute_batch(self, model, image_array, label_array):
        return self._criterion(model(image_array), label_array)


class LossEvaluatorFactory:

    def create_loss_evaluator(self, name):
        if name == 'cross_entropy':
            return CrossEntropyLossEvaluator()
        else:
            raise Exception('unsupported loss evaluator')
