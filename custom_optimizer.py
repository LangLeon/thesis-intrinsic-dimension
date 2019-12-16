import torch
from torch.optim.optimizer import Optimizer, required

"""
The whole file is basically just a copy of https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD.
The goal is to have an optimizer that is able to optimize only in a subspace.

Currently, delete everything that has to do with dampening, weight decay, etc., since we would have to think about that in detail
before!
"""


class custom_SGD(Optimizer):

    def __init__(self, params, E_split, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(custom_SGD, self).__init__(params, defaults)
        self.E_split = E_split
        self.d_dim = E_split[0].shape[1]
        self.params_d = torch.zeros(self.d_dim)

    def __setstate__(self, state):
        super(custom_SGD, self).__setstate__(state)

    def step(self, closure=None):

        grad_d = torch.zeros(self.d_dim)
        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                E = self.E_split[i]
                # Create subspace gradient
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_d += d_p.view(-1) @ E
            self.params_d -= group['lr'] * grad_d
            #import pdb; pdb.set_trace()


        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                E = self.E_split[i]

                p.data = (E @ self.params_d).reshape(p.data.shape)
