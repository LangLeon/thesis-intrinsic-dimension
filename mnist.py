import os
import errno

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


def mnist(root='./data/', batch_size=64, deterministic_split=False, seed=1, download=True):

    np.random.seed(seed) # set the seed the same as that for pytorch

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=transformation, target_transform=None,
        download=True)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=transformation, target_transform=None,
        download=True)

    if not deterministic_split:
        train_indices = np.sort(np.random.choice(60000, 50000, replace=False))
        val_indices = np.setdiff1d(np.arange(60000), train_indices)
    else:
        train_indices = np.arange(50000)
        val_indices = np.arange(50000, 60000)

    train_dataset = data.dataset.Subset(dataset, train_indices)
    val_dataset = data.dataset.Subset(dataset, val_indices)

    trainloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader
