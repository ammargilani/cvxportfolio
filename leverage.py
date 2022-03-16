from constraints.constraint import Constraint
import cvxpy as cvx
import torch


class Leverage(Constraint):

    def __init__(self, theta):
        super().__init__(is_equality=False, dim=1)
        self.theta = theta

    def evaluate(self, w, z, **kwargs):
        return cvx.norm((w + z)[0, :-1], 1) - self.theta.detach()

    def evaluate_torch(self, w, z, **kwargs):
        return torch.norm((w + z)[0, :-1], 1) - self.theta
