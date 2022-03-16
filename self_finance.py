from constraints.constraint import Constraint
import cvxpy as cvx
import torch


class SelfFinance(Constraint):

    def __init__(self):
        super().__init__(is_equality=True, dim=1)

    def evaluate(self, z, **kwargs):
        return cvx.sum(z)

    def evaluate_torch(self, z, **kwargs):
        return torch.sum(z)
