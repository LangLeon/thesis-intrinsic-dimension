import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import numpy as np
from sklearn import random_projection
from scipy.sparse import coo_matrix

from mnist import mnist
from plotting import plot_data
from models import models
from train_helpers import train_epoch
from custom_optimizer import custom_SGD

import argparse
import csv
import os
import datetime

def log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies, subspace_training=None, d_dim=None, model=None, lr=None, seed=None, n_epochs=None, batch_size=None):
    rows = zip(epochs, train_losses,train_accuracies,val_losses,val_accuracies)

    timestamp = str(datetime.datetime.utcnow())
    file_name = "subspace_{}_d_dim_{}_model_{}_lr_{}_seed_{}_epochs_{}_batchsize_{}_{}.csv".format(subspace_training, d_dim, model, lr, seed, n_epochs, batch_size, timestamp)
    full_file_name = os.path.join("logs/", file_name)

    with open(full_file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for row in rows:
            writer.writerow(row)
    f.close()

    plot_data(full_file_name)

def to_torch(E):
    # taken from https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
    values = E.data
    indices = np.vstack((E.row, E.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = E.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def to_sparse(x):
    # taken from https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/3
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def create_random_embedding(model, d_dim, device):
    D_dim = sum(p.numel() for p in model.parameters())

    """
    dist = torch.distributions.normal.Normal(0, 1)
    E = dist.sample((D_dim, d_dim))

    """
    # Create sparse matrix. See 6.6.3 here: https://scikit-learn.org/stable/modules/random_projection.html#sparse-random-matrix
    transformer = random_projection.SparseRandomProjection()
    E = transformer._make_random_matrix(D_dim, d_dim)
    E = coo_matrix(E)
    E = to_torch(E)

    # normalization of columns ---> obtain approximately orthonormal vectors, since high-dimensional!
    En = torch.norm(E, p=2, dim=0)
    E = E.div(En.expand_as(E))

    # Split E into one component for each parameter, i.e. tensor, in the model
    params = list(model.parameters())
    E_split = []
    pointer = 0
    for param in params:
        size = len(param.view(-1))
        E_split.append(E[pointer:pointer+size])
        pointer=pointer+size

    assert len(E_split) == len(params), "E_split does not have the same number of components as params!"
    for i in range(len(params)):
        assert params[i].numel() == E_split[i].shape[0], "E_split[i] has the wrong shape!"

    for i in range(len(E_split)):
        E_split[i] = to_sparse(E_split[i]).to(device)

    E_split_transpose = [E.transpose(0,1) for E in E_split]
    return E_split, E_split_transpose

def train_model_once(seed, batch_size, model, subspace_training, d_dim, lr, n_epochs, print_freq, print_prec, device):
    torch.manual_seed(seed)

    train_loader, val_loader, _ = mnist(batch_size = batch_size)
    model = models[model]().to(device)
    criterion = nn.CrossEntropyLoss()

    if subspace_training:
        E_split, E_split_transpose = create_random_embedding(model, d_dim, device) # random embedding R^d_dim ---> R^D_dim
        optimizer = custom_SGD(model.parameters(), E_split, E_split_transpose, device, lr)
    else:
        optimizer = SGD(model.parameters(), lr)
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        print("Epoch {} start".format(epoch+1))
        train_loss, train_acc, val_loss, val_acc = train_epoch(model,train_loader,val_loader,optimizer,criterion,print_freq, print_prec, device)
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies)

    return min(train_losses), max(train_accuracies), min(val_losses), max(val_accuracies)



def main():
    train_model_once(ARGS.seed, ARGS.batch_size, ARGS.model, ARGS.subspace_training, ARGS.d_dim, ARGS.lr, ARGS.n_epochs, ARGS.print_freq, ARGS.print_prec, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='max number of epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--model', default="MLP", type=str,
                        help='the model to be tested')
    parser.add_argument('--subspace_training', default=False, action='store_true',
                        help='Whether to train in the subspace or not')
    parser.add_argument('--d_dim', default=1000, type=int,
                        help='Dimension of random subspace to be trained in')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='How often the loss and accuracy should be printed')
    parser.add_argument('--print_prec', default=2, type=int,
                        help='The precision with which to print losses and accuracy.')

    ARGS = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
