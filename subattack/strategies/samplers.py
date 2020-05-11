import torch
from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(self, device):
        self._device = device

    @abstractmethod
    def sample(sample_size):
        pass


class SphereSampler(Sampler):

    def __init__(self, shape, device, epsilon=1e-7):
        super().__init__(device)
        self._shape = shape
        self._epsilon = epsilon

    def sample(self, sample_size):
        gaussian_noise = torch.randn(
            sample_size, self._shape.numel()
        ).to(self._device)

        return (
            gaussian_noise / gaussian_noise.norm(
                dim=1, keepdim=True).clamp(self._epsilon)
        ).view(-1, *self._shape)


class BallSampler(SphereSampler):

    def __init__(self, shape, device, epsilon=1e-7):
        super().__init__(shape, device, epsilon)
        n_dim = shape.numel()
        self._ratio = torch.rand((1,)).item() ** (1/n_dim)

    def sample(self, sample_size):
        return self._ratio * super().sample(sample_size)


class SubspaceSphereSampler(Sampler):

    def __init__(self, basis, device, epsilon=1e-7):
        super().__init__(device)
        self._basis = basis.to(device)
        self._epsilon = epsilon

    def sample(self, sample_size):
        gaussian_noise = torch.randn(
            sample_size, self._basis.shape[0]
        ).to(self._device)

        weights = gaussian_noise / gaussian_noise.norm(
            dim=1, keepdim=True).clamp(self._epsilon)
        return (
            weights @ self._basis.view(self._basis.shape[0], -1)
        ).view(sample_size, *self._basis.shape[1:])


class SubspaceBallSampler(SubspaceSphereSampler):

    def __init__(self, basis, device, epsilon=1e-7):
        super().__init__(basis, device, epsilon)
        n_dim = basis.shape[0]
        self._ratio = torch.rand((1,)).item() ** (1/n_dim)

    def sample(self, sample_size):
        return self._ratio * super().sample(sample_size)


class SamplerDecorator(Sampler):

    def __init__(self, sampler):
        self._sampler = sampler

    @abstractmethod
    def sample(self, sample_size):
        pass


class SamplerWithOutputDevice(SamplerDecorator):

    def __init__(self, sampler, output_device):
        super().__init__(sampler)
        self._output_device = output_device

    def sample(self, sample_size):
        return self._sampler.sample(sample_size).to(self._output_device)


class SamplerFactory:

    def create_sampler(
            self, name,
            shape=None, basis=None,
            device='cpu'
    ):
        if name == 'sphere':
            assert shape is not None
            sampler = SphereSampler(shape, device)
        elif name == 'ball':
            assert shape is not None
            sampler = BallSampler(shape, device)
        elif name == 'subspace_sphere':
            assert basis is not None
            sampler = SubspaceSphereSampler(basis, device)
        elif name == 'subspace_ball':
            assert basis is not None
            sampler = SubspaceBallSampler(basis, device)
        else:
            raise Exception('unsupported batch sampler')
        return sampler


class SamplerWithOutputDeviceFactory(SamplerFactory):

    def create_sampler(
            self, name,
            shape=None, basis=None,
            device='cpu',
            output_device='cpu',
    ):
        sampler = super().create_sampler(
            name, shape, basis, device
        )
        return SamplerWithOutputDevice(sampler, output_device)
