import matplotlib.pyplot as plt
import csv
import os
import argparse
import pandas as pd
import numpy as np


def log_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies, ARGS):
    rows = zip(epochs, train_losses,train_accuracies,val_losses,val_accuracies)

    file_name = "d_dim_{}_lr_{}_seed_{}_epochs_{}_batchsize_{}.csv".format(ARGS.d_dim, ARGS.lr, ARGS.seed, ARGS.n_epochs, ARGS.batch_size)

    subspace_training = "subspace_training" if ARGS.subspace_training else "no_subspace_training"
    model = ARGS.model

    full_file_name = os.path.join("logs/" + subspace_training + "/" + model + "/" + ARGS.timestamp, file_name)
    columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    if ARGS.ddim_vs_acc:
        if ARGS.x_axis == "d_dim":
            columns[0] = "d_dim"

    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)
    with open(full_file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
    f.close()

    plot_data(full_file_name, ARGS.ddim_vs_acc, ARGS.x_axis)


def plot_data(csv_path, ddim_vs_acc, x_axis):
    filename, _ = os.path.splitext(csv_path)
    filename = filename.replace("logs", "plots")
    print(filename)
    plot_path = filename + '.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    # plotting adapted from https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    colors = ["tab:orange", "tab:red", "tab:cyan", "tab:blue"]


    # all plots belonging to the first axis
    if not ddim_vs_acc:
        ax1.set_xlabel("Epoch")
        identifier = "epoch"
    else:
        if x_axis == "d_dim":
            ax1.set_xlabel("d_dim")
            identifier = "d_dim"
        else:
            ax1.set_xlabel("Epoch")
            identifier = "epoch"
    ax1.set_ylabel("Loss", color=colors[1])


    ax1.plot(df[identifier], df['train_loss'], label="train_loss", color=colors[0])
    ax1.plot(df[identifier], df['val_loss'], label="val_loss", color=colors[1])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis="y", labelcolor=colors[1])
    ax1.legend(handles, labels, loc='upper left')

    # all plots belonging to the second axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Accuracy", color=colors[3])
    ax2.set_ylim(0,1)
    ax2.set_yticks(np.arange(0,1.1,0.1))

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
