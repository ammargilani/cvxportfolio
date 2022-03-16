import torch


def neg_ud(x):
    return -torch.minimum(x, torch.zeros_like(x))


def quad_form_ud(x, Q):
    return x.T @ Q @ x


def is_full_rank(x):
    return torch.linalg.matrix_rank(x) == min(x.shape)
