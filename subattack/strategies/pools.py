import torch
from abc import ABC, abstractmethod


class PoolFetcher(ABC):

    def __init__(self, loader, max_size, device):
        self._loader = loader
        self._max_size = max_size
        self._device = device

    @abstractmethod
    def fetch(self):
        pass


class RandomPoolWithLabelsFetcher(PoolFetcher):

    def fetch(self):
        count = 0
        image_collection = []
        label_collection = []
        for image_array, label_array in self._loader:

            image_array = image_array.to(self._device)
            label_array = label_array.to(self._device)

            arrival_size = image_array.shape[0]
            if arrival_size + count > self._max_size:
                image_collection.append(
                    image_array[: self._max_size-count]
                )
                label_collection.append(
                    label_array[: self._max_size-count]
                )
                break
            else:
                image_collection.append(image_array)
                label_collection.append(label_array)
                count += arrival_size
        return torch.cat(image_collection), torch.cat(label_collection)


class RandomPoolFetcher(PoolFetcher):
    def __init__(self, loader, max_size, device):
        self._random_pool_with_labels_fetcher = RandomPoolWithLabelsFetcher(
            loader, max_size, device)

    def fetch(self):
        result, _ = self._random_pool_with_labels_fetcher.fetch()
        return result


class PoolFetcherFactory:

    def create_pool_fetcher(self, name, loader, max_size, device):
        if name == 'random':
            return RandomPoolFetcher(loader, max_size, device)
        elif name == 'random_with_labels':
            return RandomPoolWithLabelsFetcher(loader, max_size, device)
        else:
            raise Exception('unsupported pool fetcher')
