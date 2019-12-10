import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from mnist import mnist

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out

def train_batch(model, batch, optimizer, loss_function):
    image, label = batch
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

    model.eval()
    val_loss, val_acc = epoch_iter(model, val_loader, optimizer, loss_function)
    print("Epoch over. val_loss: {}; val_accuracy: {} \n".format(val_loss, val_acc))



n_epochs = 3
batch_size = 64
learning_rate = 0.001
# momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader, val_loader, test_loader = mnist(batch_size = batch_size)

model = Baseline()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    print("Epoch {} start".format(epoch+1))
    train_epoch(model, train_loader, val_loader, optimizer, criterion)
