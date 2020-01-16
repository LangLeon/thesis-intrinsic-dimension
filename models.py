import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from e2cnn import gspaces
from e2cnn import nn as nn2

from torchvision.transforms import Pad


pad = Pad((0, 0, 1, 1), fill=0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    # Adapted from https://github.com/activatedgeek/LeNet-5
    """
    Input - 1x28x28
    C1 - 6@24x24 (5x5 kernel)
    relu1
    S1 - 6@12x12 (2x2 kernel, stride 2) Subsampling
    C2 - 16@8x8 (5x5 kernel)
    relu2
    S2 - 16@4x4 (2x2 kernel, stride 2) Subsampling
    C3 - 120@1x1 (4x4 kernel)
    relu3
    F4 - 84
    relu4
    F5 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu3', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(84, 10)),
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class RegularLeNet5(torch.nn.Module):
    """
    Equivariant version of LeNet5.

    Adapted from Lenet:
    - C8 replaces Id, i.e. the trivial rotation group is replaced by rotations by 45% angles.
    - Consequently, the regular representation has 8 internal channels. We work with the same number of regular feature
      spaces in each layer, and thus have, say, 6*8 = 48 effective channels in the second layer (this preserves the number of parameters)
    - MaxPool2d is replaced by PointwiseMaxPool
    - relu is replaced by the equivariant relu (also operates pointwise)
    - We do a group max pooling before the fully connected layer.

    Input - 1x28x28 triv
    C1 - 6@24x24 reg (5x5 kernel)
    relu1
    S1 - 6@12x12 reg (2x2 kernel, stride 2) Subsampling
    C2 - 16@8x8 reg (5x5 kernel)
    relu2
    S2 - 16@4x4 reg (2x2 kernel, stride 2) Subsampling
    C3 - 120@1x1 reg (4x4 kernel)
    relu3
    S3 - 120@1x1 triv (group pooling)
    F4 - 84
    relu4
    F5 - 10 (Output)
    """
    def __init__(self, n_classes=10):

        super(RegularLeNet5, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = nn2.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])

        # block 1
        self.C1 = nn2.R2Conv(in_type, out_type, kernel_size=5)
        self.relu1 = nn2.ReLU(out_type)
        self.S1 = nn2.PointwiseMaxPool(out_type, kernel_size=2, stride=2)


        in_type=self.S1.out_type
        out_type = nn2.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])

        # block 2
        self.C2 = nn2.R2Conv(in_type, out_type, kernel_size=5)
        self.relu2 = nn2.ReLU(out_type)
        self.S2 = nn2.PointwiseMaxPool(out_type, kernel_size=2, stride=2)

        in_type = self.S2.out_type
        out_type =  nn2.FieldType(self.r2_act, 120*[self.r2_act.regular_repr])

        # block 3
        self.C3 = nn2.R2Conv(in_type, out_type, kernel_size=4)
        self.relu3 = nn2.ReLU(out_type)
        self.S3 = nn2.GroupPooling(out_type)


        # block 4: Fully connected
        self.F4 = nn.Linear(120, 84)
        self.F5 = nn.Linear(84, 10)


        # Complete model
        self.conv = nn2.SequentialModule(
            self.C1,
            self.relu1,
            self.S1,
            self.C2,
            self.relu2,
            self.S2,
            self.C3,
            self.relu3,
            self.S3, # This group pooling operation was not present in the original LeNet CNN.
        )

        self.fully = nn.Sequential(
            self.F4,
            nn.ReLU(),
            self.F5
        )

    def forward(self, input: torch.Tensor):

        x = nn2.GeometricTensor(input, self.input_type)
        x = self.conv(x)
        x = x.tensor.reshape(-1, 120)

        x = self.fully(x)

        return x


models = {"MLP": MLP, "lenet": LeNet5, "reg_lenet": RegularLeNet5}
