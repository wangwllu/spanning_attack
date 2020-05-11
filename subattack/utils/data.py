import json

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader


# imagenet class index
with open('subattack/json/imagenet_class_index.json') as file:
    label_to_idx = json.load(file)
    idx_to_label = [label_to_idx[str(k)][1] for k in range(len(label_to_idx))]


def generate_1by1(loader, device='cpu'):
    for image_array, label_array in loader:
        image_array, label_array = (
            image_array.to(device), label_array.to(device)
        )
        for image, label in zip(image_array, label_array):
            yield image, label


def get_loaders(data_dir, size_manifold, size_attack, num_workers, batch_size):

    val_dir = os.path.join(data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    manifold_dataset, attack_dataset = random_split(
        val_dataset, [size_manifold, size_attack])

    manifold_loader = DataLoader(
        manifold_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    attack_loader = DataLoader(
        attack_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return manifold_loader, attack_loader


def get_subspace_loader(data_dir, num_workers, batch_size):
    subspace_dataset = datasets.ImageFolder(data_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))
    return DataLoader(
        subspace_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
