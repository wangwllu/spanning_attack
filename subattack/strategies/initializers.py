import torch
from abc import ABC, abstractmethod

from subattack.strategies.samplers import Sampler
from subattack.strategies.samplers import SamplerFactory


class Initializer(ABC):

    def __init__(self, device):
        self._device = device

    @abstractmethod
    def initialize(self):
        pass


class ZeroInitializer(Initializer):

    def __init__(self, shape, device):
        super().__init__(device)
        self._shape = shape

    def initialize(self):
        return torch.zeros(self._shape).to(self._device)


class InitializerBySampling(Initializer):

    def __init__(self, radius, sampler: Sampler):
        self._radius = radius
        self._sampler = sampler

    def initialize(self):
        return self._radius * self._sampler.sample(1).squeeze(0)


class InitializerDecorator(Initializer):

    def __init__(self, initializer):
        self._initializer = initializer

    @abstractmethod
    def initialize(self):
        pass


class InitializerWithOutputDevice(InitializerDecorator):

    def __init__(self, initializer, output_device):
        super().__init__(initializer)
        self._output_device = output_device

    def initialize(self):
        return self._initializer.initialize().to(self._output_device)


class InitializerFactory:

    def create_initializer(
            self, name, radius=None,
            shape=None, basis=None, device='cpu'
    ):
        if name == 'zero':
            assert shape is not None
            return ZeroInitializer(shape, device)
        else:
            assert radius is not None
            sampler_factory = SamplerFactory()
            sampler = sampler_factory.create_sampler(
                name=name, shape=shape, basis=basis, device=device
            )
            return InitializerBySampling(radius, sampler)


class InitializerWithOutputDeviceFactory(InitializerFactory):

    def create_initializer(
            self, name, radius=None,
            shape=None, basis=None, device='cpu', output_device='cpu'
    ):
        initializer = super().create_initializer(
            name, radius, shape, basis, device
        )
        return InitializerWithOutputDevice(initializer, output_device)
