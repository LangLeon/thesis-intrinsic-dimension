import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from mnist import mnist
import argparse

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out

def train_batch(model, batch, optimizer, loss_function):
    image, label = batch
    image.to(device)
    label.to(device)
    optimizer.zero_grad()
    prediction = model(image)
    loss = torch.sum(loss_function(prediction, label))
    if model.training:
        loss.backward()
        optimizer.step()
    accuracy = (torch.argmax(prediction, 1) == label).sum().float() / prediction.shape[0]
    return loss.item(), accuracy.item()

def epoch_iter(model, data, optimizer, loss_function):
    avg_loss = 0
    t = 0
    total_loss = 0
    total_accuracy = 0

    for (x, label) in data:

        loss, accuracy = train_batch(model, (x, label), optimizer, loss_function)
        if t % 20 == 0:
            print("Batch: {}; loss: {}; acc: {}".format(t, round(loss,2), round(accuracy,2)))
        total_loss += loss
        total_accuracy += accuracy
        t+= 1

    loss_avg = total_loss / t
    accuracy_avg = total_accuracy / t

    return loss_avg, accuracy_avg

def train_epoch(model, train_loader, val_loader, optimizer, loss_function):
    model.train()
    train_loss, train_acc = epoch_iter(model, train_loader, optimizer, loss_function)
    print("Train Epoch over. train_loss: {}; train_accuracy: {} \n".format(train_loss, train_acc))

    model.eval()
    val_loss, val_acc = epoch_iter(model, val_loader, optimizer, loss_function)
    print("Val Epoch over. val_loss: {}; val_accuracy: {} \n".format(val_loss, val_acc))
    return train_loss, train_acc, val_loss, val_acc


def main():

    torch.manual_seed(ARGS.seed)

    train_loader, val_loader, _ = mnist(batch_size = ARGS.batch_size)

    model = Baseline().to(device)
    optimizer = torch.optim.SGD(model.parameters(), ARGS.lr)

    criterion = nn.CrossEntropyLoss()


    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(ARGS.n_epochs):
        print("Epoch {} start".format(epoch+1))
        train_loss, train_acc, val_loss, val_acc = train_epoch(model, train_loader, val_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    print(train_losses, train_accuracies, val_losses, val_accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='max number of epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='max number of epochs')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='max number of epochs')
    parser.add_argument('--seed', default=1, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
