import matplotlib.pyplot as plt
import csv
import os
import argparse
import pandas as pd
import datetime


def log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies, ARGS):
    rows = zip(epochs, train_losses,train_accuracies,val_losses,val_accuracies)

    timestamp = str(datetime.datetime.utcnow())
    file_name = "subspace_{}_d_dim_{}_model_{}_lr_{}_seed_{}_epochs_{}_batchsize_{}_{}.csv".format(ARGS.subspace_training, ARGS.d_dim, ARGS.model, ARGS.lr, ARGS.seed, ARGS.n_epochs, ARGS.batch_size, timestamp)
    full_file_name = os.path.join("logs/", file_name)
    columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    if ARGS.ddim_vs_acc:
        columns[0] = "d_dim"


    with open(full_file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
    f.close()

    plot_data(full_file_name, ARGS.ddim_vs_acc)


def plot_data(csv_path, ddim_vs_acc):
    filename, _ = os.path.splitext(csv_path)
    filename = filename.replace("logs", "plots")
    print(filename)
    plot_path = filename + '.png'

    df = pd.read_csv(csv_path)

    # plotting adapted from https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    colors = ["tab:orange", "tab:red", "tab:cyan", "tab:blue"]


    # all plots belonging to the first axis
    if not ddim_vs_acc:
        ax1.set_xlabel("Epoch")
        identifier = "epoch"
    else:
        ax1.set_xlabel("d_dim")
        identifier = "d_dim"
    ax1.set_ylabel("Loss", color=colors[1])


    ax1.plot(df[identifier], df['train_loss'], label="train_loss", color=colors[0])
    ax1.plot(df[identifier], df['val_loss'], label="val_loss", color=colors[1])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis="y", labelcolor=colors[1])
    ax1.legend(handles, labels, loc='upper left')

    # all plots belonging to the second axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Accuracy", color=colors[3])

    ax2.plot(df[identifier], df['train_accuracy'], label="train_acc", color=colors[2])
    ax2.plot(df[identifier], df['val_accuracy'], label="val_acc", color=colors[3])

    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.tick_params(axis="y", labelcolor=colors[3])
    ax2.legend(handles2, labels2, loc='lower left')

    # saving of figure
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='Path to the csv with metrics.')
    args = parser.parse_args()

    plot_data(args.path)