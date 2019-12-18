import torch
from torch.optim.optimizer import Optimizer, required

"""
The whole file is basically just a copy of https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD.
The goal is to have an optimizer that is able to optimize only in a subspace.

Currently, delete everything that has to do with dampening, weight decay, etc., since we would have to think about that in detail
before!
"""


class custom_SGD(Optimizer):

    def __init__(self, params, E_split, E_split_transpose, device, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(custom_SGD, self).__init__(params, defaults)
        assert len(self.param_groups) == 1, "The optimizer can currently only deal with one parameter group"
        self.E_split = E_split
        self.E_split_transpose = E_split_transpose
        self.d_dim = E_split[0].shape[1]
        self.params_d = torch.zeros(self.d_dim)
        self.device = device

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
                grad_d += torch.sparse.mm(self.E_split_transpose[i], d_p.view(-1,1)).view(-1) # d_p.view(-1).data @ self.E_split[i].data
            self.params_d.data.add_(-group['lr'],grad_d.view(-1))


        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                #import pdb; pdb.set_trace()
                p.data = torch.sparse.mm(self.E_split[i],self.params_d.view(-1,1)).reshape(p.data.shape)
