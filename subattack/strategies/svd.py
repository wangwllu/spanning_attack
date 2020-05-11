from abc import ABC, abstractmethod
import torch
import math
import numpy as np


class SvdBasisFetcher(ABC):

    def __init__(self, pool, start, end):
        assert end - start <= pool.shape[0]

        self._pool = pool
        self._start = start
        self._end = end

    def fetch(self):
        pool_device = self._pool.device

        basis = self._compute_svd(self._pool.view(self._pool.shape[0], -1))

        return basis[self._start:self._end].view(
            self._end-self._start, *self._pool.shape[1:]
        ).to(pool_device)

    @abstractmethod
    def _compute_svd(self, vectors):
        pass


class TorchSvdBasisFetcher(SvdBasisFetcher):

    def __init__(self, pool, start, end, device='cpu'):
        super().__init__(pool, start, end)
        self._device = device

    # memory eagar but very fast
    def _compute_svd(self, vectors):
        _, _, result = torch.svd(vectors.to(self._device))
        return result.t()


class NumpySvdBasisFetcher(SvdBasisFetcher):

    # memory economical but slow
    def _compute_svd(self, vectors):
        _, _, result = np.linalg.svd(
            vectors.cpu().numpy(), full_matrices=False
        )
        return torch.from_numpy(result)


class SvdBasisFetcherFactory:

    def create_svd_basis_fetcher(self, name, pool, subspace_size, position):
        if position == 'top':
            start = 0
            end = subspace_size
        elif position == 'bottom':
            start = pool.shape[0] - subspace_size
            end = pool.shape[0]
        elif position == 'middle':
            start = math.floor((pool.shape[0] - subspace_size)/2)
            end = start + subspace_size

        if name == 'torch':
            return TorchSvdBasisFetcher(pool, start, end)
        elif name == 'numpy':
            return NumpySvdBasisFetcher(pool, start, end)
        else:
            raise Exception('unsupported svd')
