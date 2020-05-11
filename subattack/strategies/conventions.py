import torch

from abc import ABC, abstractmethod


class Convention(ABC):

    def __init__(
        self,
        lower=0., upper=1., p_lower=0., p_upper=255.,
        shape=torch.Size((3, 224, 224))
    ):
        self._lower = lower
        self._upper = upper
        self._p_lower = p_lower
        self._p_upper = p_upper
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def unit(self):
        return (self._upper - self._lower) / (self._p_upper - self._p_lower)

    @abstractmethod
    def project(self, image):
        pass

    def _clamp(self, image):
        return image.clamp(self._lower, self._upper)

    def _pixel_to_normalized(self, p_image):
        return (p_image - self._p_lower) / (self._p_upper - self._p_lower) * (
            self._upper - self._lower) + self._lower

    def _normalized_to_pixel(self, image):
        return ((image - self._lower) / (self._upper - self._lower) * (
            self._p_upper - self._p_lower) + self._p_lower).round()

    def _legitimize(self, image):
        return self._pixel_to_normalized(self._normalized_to_pixel(image))


class ContinuousPixelConvention(Convention):

    def project(self, image):
        return self._clamp(image)


class DiscretePixelConvention(Convention):

    def project(self, image):
        return self._legitimize(self._clamp(image))


class ConventionFactory:

    def create_convention(self, name):
        if name == 'continuous':
            return ContinuousPixelConvention()
        elif name == 'discrete':
            return DiscretePixelConvention()
        else:
            raise Exception('unsupported convention')
