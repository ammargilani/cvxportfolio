from constraints.constraint import Constraint
import cvxpy as cvx
import numpy as np


class NoTrade(Constraint):

    def __init__(self):
        super().__init__(is_equality=True)
        self.zero_idxs = None

    def evaluate(self, z, idxs, **kwargs):
        idxs = np.append(idxs, [z.shape[1] - 1])
        self.zero_idxs = np.delete(np.arange(z.shape[1]), idxs).reshape(1, -1)
        self.dim = self.zero_idxs.shape
        return z[0, self.zero_idxs]

    def evaluate_torch(self, z, **kwargs):
        return z[0, self.zero_idxs]
