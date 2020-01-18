import os
import errno

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


class toFloat:
    def __init__(self):
        pass

    def __call__(self, img):
        return img.float()


def mnist(root='./data/', batch_size=64, download=True):

    transformation = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=transformation, target_transform=None,
        download=True)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=transformation, target_transform=None,
        download=True)

    train_dataset = data.dataset.Subset(dataset, np.arange(40000))
    val_dataset = data.dataset.Subset(dataset, np.arange(40000, 50000))

    trainloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader
