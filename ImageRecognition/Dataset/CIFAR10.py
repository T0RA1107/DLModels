import torch
from torchvision import datasets

def dataloader(batch_size, transform_train, transform_valid):
    dataloader_train = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train),
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_valid = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform_valid),
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader_train, dataloader_valid