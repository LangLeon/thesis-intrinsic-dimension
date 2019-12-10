import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
# momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)



test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        out_transform = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        out = self.out_transform(self.fc3(x))
        return F.relu(self.conv2(x))

model = Baseline()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

def train_batch(model, batch, optimizer, loss_function):
    images, labels = batch
    optimizer.zero_grad()
    probs = model(images)
    loss = torch.sum(loss_function(probs, labels))
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, )
