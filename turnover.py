from constraints.constraint import Constraint
import cvxpy as cvx
import torch


class Turnover(Constraint):

    def __init__(self, theta):
        super().__init__(is_equality=False, dim=1)
        self.theta = theta

    def evaluate(self, z, **kwargs):
        return cvx.norm(z[0, :-1], 1) - self.theta.detach()

    def evaluate_torch(self, w, z, **kwargs):
        return torch.norm(z[0, :-1], 1) - self.theta
