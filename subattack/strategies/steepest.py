from abc import ABC, abstractmethod


class SteepestGradientTransformer(ABC):

    @abstractmethod
    def transform(self, gradient):
        pass


class L2SteepestGradientTransformer(SteepestGradientTransformer):

    def __init__(self, epsilon=1e-7):
        self._epsilon = epsilon

    def transform(self, gradient):
        return gradient / gradient.norm().clamp(min=self._epsilon)


class LinfSteepestGradientTransformer(SteepestGradientTransformer):

    def transform(self, gradient):
        return gradient.sign()


class SteepestGradientTransformerFactory:

    def create_steepest_gradient_transformer(self, name):
        if name == 'l2':
            return L2SteepestGradientTransformer()
        elif name == 'linf':
            return LinfSteepestGradientTransformer()
        else:
            raise Exception('unsupported gradient transformer')
