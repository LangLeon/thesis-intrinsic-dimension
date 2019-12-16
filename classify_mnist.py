import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from mnist import mnist
from plotting import plot_data
from models import models
from models import Subspace_model
from train_helpers import train_epoch

import argparse
import csv
import os
import datetime

def log_results(epochs, train_loses, train_accuracies, val_losses, val_accuracies):
    rows = zip(epochs, train_losses,train_accuracies,val_losses,val_accuracies)

    timestamp = str(datetime.datetime.utcnow())
    file_name = "lr_{}_seed_{}_epochs_{}_batchsize_{}_{}.csv".format(ARGS.lr, ARGS.seed, ARGS.n_epochs, ARGS.batch_size, timestamp)
    full_file_name = os.path.join("logs/", file_name)

    with open(full_file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for row in rows:
            writer.writerow(row)
    f.close()

    plot_data(full_file_name)


def create_random_embedding(model, d_dim):
    D_dim = sum(p.numel() for p in model.parameters() if p.requires_grad) # Number of trainable parameters
    dist = torch.distributions.normal.Normal(0, 1)
    E = dist.sample((D_dim, d_dim))

    # normalization of columns ---> Approximately orthonormal vectors, since high-dimensional!
    En = torch.norm(E, p=2, dim=0).detach()
    E = E.div(En.expand_as(E))

    return E



def main():
    torch.manual_seed(ARGS.seed)

    train_loader, val_loader, _ = mnist(batch_size = ARGS.batch_size)
    model = models[ARGS.model]().to(device)
    criterion = nn.CrossEntropyLoss()

    if ARGS.subspace_training:
        E = create_random_embedding(model, ARGS.d_dim) # random embedding R^d_dim ---> R^D_dim
        model = Subspace_model(model, E)

    optimizer = torch.optim.SGD(model.parameters(), ARGS.lr)
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(ARGS.n_epochs):
        print("Epoch {} start".format(epoch+1))
        train_loss, train_acc, val_loss, val_acc = train_epoch(model,train_loader,val_loader,optimizer,criterion,device)
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,
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

    ARGS = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
