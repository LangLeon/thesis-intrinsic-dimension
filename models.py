import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from e2cnn import gspaces
from e2cnn import nn as nn2

from torchvision.transforms import Pad

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x   = F.relu(self.fc2(x))
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


class RegLenet(torch.nn.Module):
    """
    Equivariant version of LeNet5.

    Adapted from Lenet:
    - C8 replaces Id, i.e. the trivial rotation group is replaced by rotations by 45% angles.
    - Consequently, the regular representation has 8 internal channels. We work with the same number of regular feature
      spaces in each layer, and thus have, say, 6*8 = 48 effective channels in the second layer (this preserves the number of parameters)
    - MaxPool2d is replaced by PointwiseMaxPoolAntialiased
    - relu is replaced by the equivariant relu (also operates pointwise)
    - We do a group max pooling before the fully connected layer.

    !!!Doc might be wrong!!!
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

        super(RegLenet, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = nn2.FieldType(self.r2_act, 4*[self.r2_act.regular_repr])

        # block 1
        self.C1 = nn2.R2Conv(in_type, out_type, kernel_size=5)
        self.relu1 = nn2.ReLU(out_type)
        self.S1 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)


        in_type=self.S1.out_type
        out_type = nn2.FieldType(self.r2_act, 15*[self.r2_act.regular_repr])

        # block 2
        self.C2 = nn2.R2Conv(in_type, out_type, kernel_size=5)
        self.relu2 = nn2.ReLU(out_type)
        self.S2 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

        in_type = self.S2.out_type
        out_type =  nn2.FieldType(self.r2_act, 55*[self.r2_act.regular_repr])

        # block 3
        self.C3 = nn2.R2Conv(in_type, out_type, kernel_size=4)
        self.relu3 = nn2.ReLU(out_type)
        self.S3 = nn2.GroupPooling(out_type)


        # block 4: Fully connected
        self.F4 = nn.Linear(55, 84)
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
        x = x.tensor.squeeze()
        x = self.fully(x)

        return x


class RegLenet2(torch.nn.Module):
    """
    Equivariant version of LeNet5.

    Less downsampling!
    """
    def __init__(self, n_classes=10):

        super(RegLenet2, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = nn2.FieldType(self.r2_act, 4*[self.r2_act.regular_repr])

        # block 1
        self.C11 = nn2.R2Conv(in_type, out_type, kernel_size=3)
        self.relu11 = nn2.ReLU(out_type)
        self.C12 = nn2.R2Conv(out_type, out_type, kernel_size=3)
        self.relu12 = nn2.ReLU(out_type)
        self.S1 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)


        in_type=self.S1.out_type
        out_type = nn2.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])

        # block 2
        self.C21 = nn2.R2Conv(in_type, out_type, kernel_size=3)
        self.relu21 = nn2.ReLU(out_type)
        self.C22 = nn2.R2Conv(out_type, out_type, kernel_size=3)
        self.relu22 = nn2.ReLU(out_type)
        self.S2 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)


        in_type = self.S2.out_type
        out_type =  nn2.FieldType(self.r2_act, 20*[self.r2_act.regular_repr])

        # block 3
        self.C3 = nn2.R2Conv(in_type, out_type, kernel_size=3)
        self.relu3 = nn2.ReLU(out_type)
        self.S31 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)
        self.S32 = nn2.GroupPooling(out_type)


        # block 4: Fully connected
        self.F4 = nn.Linear(20, 84)
        self.F5 = nn.Linear(84, 10)


        # Complete model
        self.conv = nn2.SequentialModule(
            self.C11,
            self.relu11,
            self.C12,
            self.relu12,
            self.S1,
            self.C21,
            self.relu21,
            self.C22,
            self.relu22,
            self.S2,
            self.C3,
            self.relu3,
            self.S31,
            self.S32
        )

        self.fully = nn.Sequential(
            self.F4,
            nn.ReLU(),
            self.F5
        )

    def forward(self, input: torch.Tensor):

        x = nn2.GeometricTensor(input, self.input_type)
        x = self.conv(x)
        x = x.tensor.squeeze()
        x = self.fully(x)

        return x


class RegLenet3(torch.nn.Module):
    """
    invariant in the end!
    """
    def __init__(self, n_classes=10):

        super(RegLenet3, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = nn2.FieldType(self.r2_act, 4*[self.r2_act.regular_repr])

        # block 1
        self.C11 = nn2.R2Conv(in_type, out_type, kernel_size=3)
        self.relu11 = nn2.ReLU(out_type)
        self.C12 = nn2.R2Conv(out_type, out_type, kernel_size=3)
        self.relu12 = nn2.ReLU(out_type)
        self.S1 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)


        in_type=self.S1.out_type
        out_type = nn2.FieldType(self.r2_act, 5*[self.r2_act.regular_repr])

        # block 2
        self.C21 = nn2.R2Conv(in_type, out_type, kernel_size=3)
        self.relu21 = nn2.ReLU(out_type)
        self.C22 = nn2.R2Conv(out_type, out_type, kernel_size=3)
        self.relu22 = nn2.ReLU(out_type)
        self.S2 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)



        # block 3
        self.C3 = nn.Conv2d(40, 60, kernel_size=(4, 4))
        self.relu3 = nn.ReLU()


        # block 4: Fully connected
        self.F4 = nn.Linear(60, 50)
        self.F5 = nn.Linear(50, 10)


        # Complete model
        self.conv_equi = nn2.SequentialModule(
            self.C11,
            self.relu11,
            self.C12,
            self.relu12,
            self.S1,
            self.C21,
            self.relu21,
            self.C22,
            self.relu22,
            self.S2
        )

        self.conv_standard = nn.Sequential(
            self.C3,
            self.relu3
        )

        self.fully = nn.Sequential(
            self.F4,
            nn.ReLU(),
            self.F5
        )

    def forward(self, input: torch.Tensor):

        x = nn2.GeometricTensor(input, self.input_type)
        x = self.conv_equi(x)
        x = x.tensor
        x = self.conv_standard(x)
        x = x.squeeze()
        x = self.fully(x)

        return x


class Table13Model(torch.nn.Module):

    def __init__(self, n_classes=10):

        super(Table13Model, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type


        # block 11
        out_type = nn2.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block11 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=7, padding=1, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )


        # block 12
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block12 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=2, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        self.S1 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

        # block 21
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block21 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=2, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        # block 22
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block22 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=2, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        self.S2 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

        # block 31
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block31 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=2, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        # block 32
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block32 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=0, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        self.S31 = nn2.GroupPooling(out_type)
        out_type = nn2.FieldType(self.r2_act, 64*[self.r2_act.trivial_repr])
        self.S32 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)


        # block 4: Fully connected
        self.F4 = nn.Linear(64, 64)
        self.F5 = nn.Linear(64, 10)


        # full conv part
        self.conv = nn2.SequentialModule(
            self.block11,
            self.block12,
            self.S1,
            self.block21,
            self.block22,
            self.S2,
            self.block31,
            self.block32,
            self.S31,
            self.S32
        )

        # fully part
        self.fully = nn.Sequential(
            self.F4,
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            self.F5
        )

    def forward(self, input: torch.Tensor):
        x = nn2.GeometricTensor(input, self.input_type)
        x = self.conv(x)
        x = x.tensor.squeeze()
        x = self.fully(x)

        return x


class Table13ModelSlim(torch.nn.Module):

    def __init__(self, N, flips, n_classes=10):

        super(Table13ModelSlim, self).__init__()

        self.N = N
        self.flips = flips

        if self.flips:
            self.r2_act = gspaces.FlipRot2dOnR2(N=self.N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=self.N)

        # The channel scaling factor is 1 for D_16. It scales the number of channels such that
        # the model overall has roughly as many parameters as the D_16 default choice model.
        scaling_factor = 4/(math.sqrt(self.N))
        if not self.flips:
            scaling_factor *= math.sqrt(2*1.15)
        print("\n Channel scaling factor: {}".format(scaling_factor))


        in_type = nn2.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type


        # block 11
        out_type = nn2.FieldType(self.r2_act, math.ceil(8*scaling_factor)*[self.r2_act.regular_repr])
        self.block11 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=7, padding=0, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )


        # block 12
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, math.ceil(12*scaling_factor)*[self.r2_act.regular_repr])
        self.block12 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=0, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        self.S1 = nn2.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

        # block 2 - only has one convolution, not two.
        in_type = out_type
        out_type = nn2.FieldType(self.r2_act, math.ceil(24*scaling_factor)*[self.r2_act.regular_repr])
        self.block2 = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=0, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        # restriction to trivial group = usual convolution
        if self.flips:
            self.restrict = nn2.RestrictionModule(in_type=out_type, id=(None,1))
        else:
            self.restrict = nn2.RestrictionModule(in_type=out_type, id=1)


        # New group, due to restriction!
        self.r2_act_2 = gspaces.TrivialOnR2()

        # normal conv block
        in_type = self.restrict.out_type
        out_type = nn2.FieldType(self.r2_act_2, 64*[self.r2_act_2.regular_repr])
        self.normal_conv = nn2.SequentialModule(
            nn2.R2Conv(in_type, out_type, kernel_size=5, padding=0, maximum_offset=0),
            nn2.InnerBatchNorm(out_type),
            nn2.ReLU(out_type, inplace=True)
        )

        # block 4: Fully connected
        self.F3 = nn.Linear(64, 64)
        self.F4 = nn.Linear(64, 10)


        # full conv part
        self.conv = nn2.SequentialModule(
            self.block11,
            self.block12,
            self.S1,
            self.block2
        )

        # fully part
        self.fully = nn.Sequential(
            self.F3,
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            self.F4
        )

    def forward(self, input: torch.Tensor):
        x = nn2.GeometricTensor(input, self.input_type)
        x = self.conv(x)
        x = self.restrict(x)
        x = self.normal_conv(x)
        x = x.tensor.squeeze()
        x = self.fully(x)

        return x


models = {
    "MLP": MLP,
    "lenet": LeNet5,
    "reg_lenet": RegLenet,
    "reg_lenet_2": RegLenet2,
    "reg_lenet_3": RegLenet3,
    "table13": Table13Model,
    "table13slim": Table13ModelSlim}
