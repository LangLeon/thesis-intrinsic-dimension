import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.convert_parameters import vector_to_parameters


class Subspace_model(nn.Module):
    """
    Wraps the function below. For convenience, since we also want to save parameters somewhere.
    """
    def __init__(self, model, E, params_d, params_0):
        super(Subspace_model, self).__init__()
        self.model = model
        self.E = E
        self.params_d = params_d
        self.params_0 = params_0
        self.params_D = self.E @ self.params_d + self.params_0

    def forward(self, x):
        return Subspace_model_F.apply(x, self.model, self.E, self.params_d, self.params_0)

class Subspace_model_F(torch.autograd.Function):
    """
    Wrapper module: At its core, it has a usual pytorch module like LeNet5 or an MLP.
    It additionally includes an embedding E: R^d ---> R^D in order to be able to
    train in a subspace of dimension d.
    """

    def forward(ctx, x, model, E, params_d, params_0):
        # Two things happen: params_d is transformed to be the parameters of self.model.
        # Then, the model performs the computation!

        params_D = E @ params_d + params_0
        vector_to_parameters(params_D, model.parameters())

        ctx.save_for_backward(E)
        ctx.model = model

        return model.forward(x)


    def backward(ctx, grad_out):
        E = ctx.saved_tensors
        model = ctx.model

        grad_x, grad_D = model.backward(grad_out)
        grad_d = grad_D @ self.E

        return grad_x, None, None, grad_d, None



class LeNet5(nn.Module):
    # Taken from https://github.com/activatedgeek/LeNet-5
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            #('sig7', nn.LogSoftmax(dim=-1)) Removed, since in my case, I use a loss that already includes this!
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

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


models = {"MLP": MLP, "lenet": LeNet5}
