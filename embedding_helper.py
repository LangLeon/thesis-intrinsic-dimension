import numpy as np
from sklearn import random_projection
from scipy.sparse import coo_matrix
import torch



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



def create_random_embedding(model, d_dim, device):
    D_dim = sum(p.numel() for p in model.parameters())

    """
    dist = torch.distributions.normal.Normal(0, 1)
    E = dist.sample((D_dim, d_dim))

    """
    # Create sparse matrix. See 6.6.3 here: https://scikit-learn.org/stable/modules/random_projection.html#sparse-random-matrix
    transformer = random_projection.SparseRandomProjection()
    E = transformer._make_random_matrix(D_dim, d_dim)
    E = coo_matrix(E)
    E = to_torch(E)

    # normalization of columns ---> obtain approximately orthonormal vectors, since high-dimensional!
    En = torch.norm(E, p=2, dim=0)
    E = E.div(En.expand_as(E))

    # Split E into one component for each parameter, i.e. tensor, in the model
    params = list(model.parameters())
    E_split = []
    pointer = 0
    for param in params:
        size = len(param.view(-1))
        E_split.append(E[pointer:pointer+size])
        pointer=pointer+size

    assert len(E_split) == len(params), "E_split does not have the same number of components as params!"
    for i in range(len(params)):
        assert params[i].numel() == E_split[i].shape[0], "E_split[i] has the wrong shape!"

    for i in range(len(E_split)):
        E_split[i] = to_sparse(E_split[i]).to(device)

    E_split_transpose = [E.transpose(0,1) for E in E_split]
    return E_split, E_split_transpose
