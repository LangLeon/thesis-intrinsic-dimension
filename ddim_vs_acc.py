from classify_mnist import train_model_once
from logging_helper import log_results

import argparse
import torch

import datetime


def main():
    if ARGS.model == "MLP":
        d_dims = [100, 200, 400, 600, 800, 1000]
    elif ARGS.model == "lenet":
        d_dims = [10, 25, 50, 75, 100, 125, 150, 175, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500]
    elif ARGS.model == "reg_lenet_3":
        d_dims = [10, 25, 50, 75, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500]
    else:
        d_dims = [50, 100, 200, 300, 400, 500]
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for d_dim in d_dims:
        ARGS.d_dim = d_dim
        ARGS.x_axis = "epochs"
        train_loss, train_accuracy, val_loss, val_accuracy = train_model_once(ARGS)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    ARGS.d_dim = "XXXXX"
    ARGS.x_axis = "d_dim"
    subspace_distances = len(d_dims)*[None]
    log_results(d_dims, train_losses, train_accuracies, val_losses, val_accuracies, subspace_distances, ARGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Most important settings
    parser.add_argument('--model', default="MLP", type=str,
                        help='the model to be tested')
    parser.add_argument('--N', default="16", type=int,
                        help='specifies N in C_N or D_N')
    parser.add_argument('--flips', action="store_true", default=False,
                        help='whether to also have reflections, i.e. to use D_N instead of C_N')
    parser.add_argument('--optimizer', default="SGD", type=str,
                        help='the optimizer to be used')
    # subspace_training missing, since this is the point here.
    parser.add_argument('--deterministic_split', default=False, action='store_true',
                        help='Whether train and validation set of MNIST should be split deterministically')


    # Hyperparameters
    parser.add_argument('--lr', default=1, type=float,
                        help='learning rate')
    parser.add_argument('--schedule', action="store_true", default=False,
                        help='Whether to use a schedule on the lr')
    parser.add_argument('--schedule_gamma', default=0.4, type=float,
                        help='multiplier of learning rate')
    parser.add_argument('--schedule_freq', default=10, type=int,
                        help='how often learning rate is reduced by schedule_gamma')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed')
    parser.add_argument('--n_epochs', default=50, type=int,
                        help='max number of epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    # d_dim missing, since set dynamically


    # Arguments for implementation details of subspace training
    parser.add_argument('--non_wrapped', action="store_true", default=False,
                        help='Whether or not to use the *wrapped* version of the subspace optimizer')
    parser.add_argument('--chunked', action="store_true", default=False,
                        help='Whether to chunk the sparse matrix in several smaller matrices or not.')
    parser.add_argument('--dense', action="store_true", default=False,
                        help='Whether to use a dense embedding matrix instead.')
    parser.add_argument('--parameter_correction', action="store_true", default=False,
                        help='Whether to do a parameter correction.')


    # Only changes visible results
    parser.add_argument('--print_freq', default=20, type=int,
                        help='How often the loss and accuracy should be printed')
    parser.add_argument('--print_prec', default=2, type=int,
                        help='The precision with which to print losses and accuracy.')


    ARGS = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ARGS.device = device
    ARGS.subspace_training = True
    ARGS.ddim_vs_acc = True
    ARGS.timestamp = str(datetime.datetime.utcnow().replace(microsecond=0))

    dct = vars(ARGS)
    for key in dct.keys():
        print("{} : {}".format(key, dct[key]))

    main()
