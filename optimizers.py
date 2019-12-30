import torch
from torch.optim.optimizer import Optimizer, required


class WrappedOptimizer:
    """
    Wraps an arbitrary optimizer in order to train in a subspace
    """
    def __init__(self, optimizer, E_split, E_split_transpose, device):
        self.optimizer = optimizer
        assert len(self.optimizer.param_groups) == 1, "The optimizer can currently only deal with one parameter group"
        self.E_split = E_split
        self.E_split_transpose = E_split_transpose
        self.d_dim = E_split[0].shape[1]
        self.device = device

    def __project_gradient(self):
        grad_d = torch.zeros(self.d_dim).to(self.device)

        for group in self.optimizer.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                # Create subspace gradient
                d_p = p.grad.data
                grad_d += torch.sparse.mm(self.E_split_transpose[i], d_p.view(-1,1)).view(-1)

        for group in self.optimizer.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                p.grad.data = torch.sparse.mm(self.E_split[i], grad_d.view(-1,1)).reshape(p.grad.data.shape)

    def step(self):
        self.__project_gradient()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()




class CustomSGD(Optimizer):
    """
    The whole class is basically just a copy of https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD.
    The goal is to have an optimizer that is able to optimize only in a subspace.

    Currently, delete everything that has to do with dampening, weight decay, etc.
    """

    def __init__(self, params, E_split, E_split_transpose, device, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)
        assert len(self.param_groups) == 1, "The optimizer can currently only deal with one parameter group"
        self.E_split = E_split
        self.E_split_transpose = E_split_transpose
        self.d_dim = E_split[0].shape[1]
        self.device = device
        self.params_d = torch.zeros(self.d_dim).to(device)
        self.start_params = [self.param_groups[0]['params'][i].data for i in range(len(self.param_groups[0]['params']))]


    def __setstate__(self, state):
        super(custom_SGD, self).__setstate__(state)

    def step(self):

        grad_d = torch.zeros(self.d_dim).to(self.device)

        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                assert p.grad is not None, "The optimizer currently can only deal with a full model gradient, due to the embedding E."
                # Create subspace gradient
                d_p = p.grad.data
                grad_d += torch.sparse.mm(self.E_split_transpose[i], d_p.view(-1,1)).view(-1)
            self.params_d.data.add_(-group['lr'],grad_d.view(-1))


        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                p.data = torch.sparse.mm(self.E_split[i],self.params_d.view(-1,1)).reshape(p.data.shape) + self.start_params[i]