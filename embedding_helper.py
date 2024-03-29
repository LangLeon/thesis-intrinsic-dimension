import numpy as np
from sklearn import random_projection
from scipy.sparse import coo_matrix
import torch
import math


def to_torch(E):
    # taken from https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
    values = E.data
    indices = np.vstack((E.row, E.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = E.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def to_sparse(x):
    # taken from https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/3
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def print_density(E):
    num_nonzero = (E!=0).sum().item()
    num_el = E.numel()
    fraction = num_nonzero / num_el
    print("nonzero elements in E: {}".format(num_nonzero))
    print("elements in E: {}".format(num_el))
    print("fraction nonzero: {}".format(fraction))



def create_random_embedding(model, d_dim, device, chunked, dense):
    D_dim = sum(p.numel() for p in model.parameters())

    if not dense:
        # Create sparse matrix. See 6.6.3 here: https://scikit-learn.org/stable/modules/random_projection.html#sparse-random-matrix
        transformer = random_projection.SparseRandomProjection(density=1/math.sqrt(D_dim))
        E = transformer._make_random_matrix(D_dim, d_dim)
        E = coo_matrix(E)
        E = to_torch(E)
        # normalization of columns ---> obtain approximately orthonormal vectors, since high-dimensional!
        En = torch.norm(E, p=2, dim=0)

    else:
        # create dense matrix
        dist = torch.distributions.normal.Normal(0, 1)
        E = dist.sample((D_dim, d_dim))
        # normalization of columns ---> obtain approximately orthonormal vectors, since high-dimensional!
        En = torch.norm(E, p=2, dim=0)

    E = E.div(En.expand_as(E))

    print_density(E)

    E_T = E.transpose(0,1)

    if not chunked:
        if not dense:
            return to_sparse(E).to(device), to_sparse(E_T).to(device)
        else:
            return E.to(device), E_T.to(device)

    else:
        # Split E into one component for each parameter, i.e. tensor, in the model
        params = list(model.parameters())
        E_split = []
        pointer = 0
        for param in params:
            size = param.numel()
            E_split.append(E[pointer:pointer+size].to(device))
            pointer=pointer+size

        E_split_transpose = [E.transpose(0,1).to(device) for E in E_split]

        assert len(E_split) == len(params), "E_split does not have the same number of components as params!"

        if not dense:
            for i in range(len(params)):
                E_split[i] = to_sparse(E_split[i]).to(device)
                E_split_transpose[i] = to_sparse(E_split_transpose[i]).to(device)

            return E_split, E_split_transpose

        else:
            return E_split, E_split_transpose
