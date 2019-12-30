import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from mnist import mnist
from models import models
from train_helpers import train_epoch
from optimizers import CustomSGD, WrappedOptimizer
from embedding_helper import create_random_embedding
from logging_helper import log_results

import argparse


def train_model_once(ARGS):
    torch.manual_seed(ARGS.seed)
    optimizers = {"SGD": torch.optim.SGD,
                  "RMSprop": torch.optim.RMSprop,
                  "Adam": torch.optim.Adam}

    train_loader, val_loader, _ = mnist(batch_size = ARGS.batch_size)
    model = models[ARGS.model]().to(ARGS.device)
    criterion = nn.CrossEntropyLoss()

    if ARGS.subspace_training:
        E_split, E_split_transpose = create_random_embedding(model, ARGS.d_dim, ARGS.device) # random embedding R^d_dim ---> R^D_dim
        if ARGS.non_wrapped:
            assert ARGS.optimizer == "SGD", "only SGD exists in a non-wrapped, custom version"
            optimizer = CustomSGD(model.parameters(), E_split, E_split_transpose, ARGS.device, ARGS.lr)
        else:
            optimizer = optimizers[ARGS.optimizer](model.parameters(), ARGS.lr)
            optimizer = WrappedOptimizer(optimizer, E_split, E_split_transpose, device)
    else:
        optimizer = SGD(model.parameters(), ARGS.lr)
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(ARGS.n_epochs):
        print("Epoch {} start".format(epoch+1))
        train_loss, train_acc, val_loss, val_acc = train_epoch(model,train_loader,val_loader,optimizer,criterion,ARGS.print_freq, ARGS.print_prec, ARGS.device)
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies, ARGS)

    return min(train_losses), max(train_accuracies), min(val_losses), max(val_accuracies)



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
    parser.add_argument('--optimizer', default="SGD", type=str,
                        help='the optimizer to be used')
    parser.add_argument('--subspace_training', default=False, action='store_true',
                        help='Whether to train in the subspace or not')
    parser.add_argument('--non_wrapped', action="store_true", default=False,
                        help='Whether or not to use the *wrapped* version of the subspace optimizer')
    parser.add_argument('--d_dim', default=1000, type=int,
                        help='Dimension of random subspace to be trained in')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='How often the loss and accuracy should be printed')
    parser.add_argument('--print_prec', default=2, type=int,
                        help='The precision with which to print losses and accuracy.')

    ARGS = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ARGS.device=device
    ARGS.ddim_vs_acc = False

    train_model_once(ARGS)
