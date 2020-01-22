import torch
from torch.optim.optimizer import Optimizer, required
from embedding_helper import to_torch


class WrappedOptimizer:
    """
    Wraps an arbitrary optimizer in order to train in a subspace
    """
    def __init__(self, optimizer, E, E_T, device, chunked, dense):
        self.optimizer = optimizer
        assert len(self.optimizer.param_groups) == 1, "The optimizer can currently only deal with one parameter group"
        self.E = E
        self.E_T = E_T
        if chunked:
            self.d_dim = E[0].shape[1]
        else:
            self.d_dim = E.shape[1]
        self.device = device
        self.chunked = chunked
        self.dense = dense

        params = self.optimizer.param_groups[0]['params']
        if self.chunked:
            self.start_params = [p.data for p in params]
        else:
            self.start_params = torch.cat([p.data.view(-1) for p in params])


    def __project_gradient(self):
        # Projects the gradient into the subspace.
        if self.chunked:
            grad_d = torch.zeros(self.d_dim).to(self.device)
            for group in self.optimizer.param_groups:
                params = group['params']
                for i in range(len(params)):
                    # Create subspace gradient
                    d_p = params[i].grad.data
                    if not self.dense:
                        grad_d += torch.sparse.mm(self.E_T[i], d_p.view(-1,1)).view(-1)
                    else:
                        grad_d += self.E_T[i] @ d_p.view(-1)

            for group in self.optimizer.param_groups:
                params = group['params']
                for i in range(len(params)):
                    p = params[i]
                    if not self.dense:
                        p.grad.data = torch.sparse.mm(self.E[i], grad_d.view(-1,1)).reshape(p.grad.data.shape)
                    else:
                        p.grad.data = (self.E[i] @ grad_d.view(-1)).reshape(p.grad.data.shape)

        else:
            for group in self.optimizer.param_groups:
                params = group['params']
                grad_D = torch.cat([p.grad.data.view(-1) for p in params]).view(-1,1).to(self.device)
                if not self.dense:
                    grad_d = torch.sparse.mm(self.E_T, grad_D)
                    grad_D = torch.sparse.mm(self.E, grad_d).view(-1)
                else:
                    grad_d = self.E_T @ grad_D
                    grad_D = self.E @ grad_d
                pointer = 0
                for p in params:
                    size = p.numel()
                    p.grad.data = grad_D[pointer:pointer+size].reshape(p.grad.data.shape)
                    pointer = pointer + size

    def step(self):
        self.__project_gradient()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def parameter_correction(self):
        # Does almost the same as __project_gradient, but acting on the parameter vector instead of the gradient.
        if self.chunked:
            diff_d = torch.zeros(self.d_dim).to(self.device)
            for group in self.optimizer.param_groups:
                params = group['params']
                for i in range(len(params)):
                    # Create subspace parameter vector
                    diff_p = params[i].data - self.start_params[i]
                    if not self.dense:
                        diff_d += torch.sparse.mm(self.E_T[i], diff_p.view(-1,1)).view(-1)
                    else:
                        diff_d += self.E_T[i] @ diff_p.view(-1)

            for group in self.optimizer.param_groups:
                params = group['params']
                for i in range(len(params)):
                    p = params[i]
                    if not self.dense:
                        p.data = torch.sparse.mm(self.E[i], diff_d.view(-1,1)).reshape(p.data.shape) + self.start_params[i]
                    else:
                        p.data = (self.E[i] @ diff_d.view(-1)).reshape(p.data.shape) + self.start_params[i]
        else:
            for group in self.optimizer.param_groups:
                params = group['params']
                diff_D = (torch.cat([p.data.view(-1) for p in params]) - self.start_params).view(-1,1).to(self.device)
                if not self.dense:
                    diff_d = torch.sparse.mm(self.E_T, diff_D)
                    p_D = torch.sparse.mm(self.E, diff_d).view(-1) + self.start_params
                else:
                    diff_d = self.E_T @ diff_D
                    p_D = self.E @ diff_d + self.start_params
                pointer = 0
                for p in params:
                    size = p.numel()
                    p.data = p_D[pointer:pointer+size].reshape(p.data.shape)
                    pointer = pointer + size



    def compute_subspace_distance(self):
        """
        Also computes the subspace distance, but without making any assumptions about the orthogonality of the matrix E
        """

        params = self.optimizer.param_groups[0]['params']
        target = torch.cat([p.data.view(-1) for p in params]) - self.start_params

        """
        if not self.dense:
            import pdb; pdb.set_trace()
            theta_d = torch.inverse(torch.sparse.mm(self.E_T, self.E)) @ torch.sparse.mm(self.E_T,target.view(-1,1))
        else:
        """
        if not self.dense:
            E = self.E.to_dense()
            E_T = self.E_T.to_dense()
        else:
            E = self.E
            E_T = self.E_T
        theta_d = torch.inverse(E_T @ E) @ (E_T @ target)
        theta_D = E @ theta_d

        return torch.dist(theta_D, target)


class CustomSGD(Optimizer):
    """
    The class is adapted from https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD.
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
