import torch.nn as nn
from torch.utils.data import DataLoader


def decide(s):
    return s.max(dim=1)[1]


def epoch(model, loader: DataLoader, opt=None, device='cpu'):

    total_loss, total_err = 0., 0.
    criterion = nn.CrossEntropyLoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        s = model(X)
        loss = criterion(s, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (decide(s) != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
