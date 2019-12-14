import matplotlib.pyplot as plt
import csv
import os
import argparse
import pandas as pd


def plot_data(csv_path):
    filename, _ = os.path.splitext(csv_path)
    filename = filename.replace("logs", "plots")
    print(filename)
    plot_path = filename + '.png'

    df = pd.read_csv(csv_path)

    # plotting taken from https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    colors = ["tab:orange", "tab:red", "tab:cyan", "tab:blue"]


    # all plots belonging to the first axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=colors[1])

    ax1.plot(df['epoch'], df['train_loss'], label="train_loss", color=colors[0])
    ax1.plot(df['epoch'], df['val_loss'], label="val_loss", color=colors[1])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis="y", labelcolor=colors[1])
    ax1.legend(handles, labels, loc='upper left')

    # all plots belonging to the second axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Accuracy", color=colors[3])

    ax2.plot(df['epoch'], df['train_accuracy'], label="train_acc", color=colors[2])
    ax2.plot(df['epoch'], df['val_accuracy'], label="val_acc", color=colors[3])

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
