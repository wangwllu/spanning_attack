import torch
import torch.nn as nn

import torchvision


class NormalizingHead(nn.Module):
    """A head module for normalizing imagenet"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self._mean = mean
        self._std = std

    def forward(self, input):
        channels = [(torch.unsqueeze(input[:, i], 1) -
                     self._mean[i]) / self._std[i] for i in range(3)]
        output = torch.cat(channels, 1)
        return output

    def extra_repr(self):
        return 'mean={}, std={}'.format(self._mean, self._std)


class ModelFactory:

    def create_model(self, name, device):
        model = nn.Sequential(
            NormalizingHead(),
            getattr(torchvision.models, name)(pretrained=True)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model.to(device)
