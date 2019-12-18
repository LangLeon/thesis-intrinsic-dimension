from classify_mnist import train_model_once, log_results

import argparse
import torch


def main():
    if ARGS.model == "MLP":
        d_dims = [100, 200, 400, 600, 800, 1000]
    else:
        d_dims = [50, 100, 200, 300, 400, 500]

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for d_dim in d_dims:
        train_loss, train_accuracy, val_loss, val_accuracy = train_model_once(ARGS.seed, ARGS.batch_size, ARGS.model, True, d_dim, ARGS.lr, ARGS.n_epochs, ARGS.print_freq, ARGS.print_prec, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    log_results(d_dims, train_losses, train_accuracies, val_losses, val_accuracies, True, "XXXXX", ARGS.model, ARGS.lr, ARGS.seed, ARGS.n_epochs, ARGS.batch_size)

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
    #parser.add_argument('--subspace_training', default=False, action='store_true',
    #                    help='Whether to train in the subspace or not')
    #parser.add_argument('--d_dim', default=1000, type=int,
    #                    help='Dimension of random subspace to be trained in')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='How often the loss and accuracy should be printed')
    parser.add_argument('--print_prec', default=2, type=int,
                        help='The precision with which to print losses and accuracy.')

    ARGS = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
