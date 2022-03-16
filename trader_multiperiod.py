import warnings

import numpy as np
import cvxpy as cvx
from tqdm import tqdm

from generic import warn
from trader import Trader


class MultiPeriodTrader(Trader):
    def forward(self, returns_vector, covariance, active_set, fake_trading_idx):
        fake_trading_idx = [0] + fake_trading_idx
        var_vector = []
        objective = 0.
        constraints = []
        w = self.w.copy()
        for i, returns in enumerate(returns_vector):
            if i not in fake_trading_idx:
                z = np.zeros(w.shape)
                objective += self.eval_objective_normalized(z, returns, covariance, w=w)
            else:
                z = cvx.Variable(w.shape)
                var_vector.append(z)
                w_plus = w + z
                objective += self.eval_objective_normalized(z, returns, covariance, w=w)
                constraints += [constraint.generate(
                    w=w, z=z, w_plus=w_plus,
                    idxs=active_set
                ) for constraint in self.constraints]
                w = w_plus
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        # No trading if there's no solution
        if prob.status == 'infeasible':
            warn('No solution!')
            self.z = np.zeros(z.shape)
        else:
            self.z = var_vector[0].value

    def trade(
            self, returns_vector, covariance, active_set, fake_trading_idx=[5, 25]):
        self.forward(returns_vector, covariance, active_set, fake_trading_idx)
        self.propagate(self.z, returns_vector[0])

    def trade_vector(self, returns_vectors, covariance_vectors, active_set_vectors,
                     fake_trading_idx=None):
        if fake_trading_idx is None:
            for r_vector, S_vector, idx_vector in \
                    tqdm(zip(returns_vectors, covariance_vectors, active_set_vectors),
                         total=len(returns_vectors)):
                self.trade(r_vector, S_vector, idx_vector)
        else:
            for r_vector, S_vector, idx_vector in \
                    tqdm(zip(returns_vectors, covariance_vectors, active_set_vectors),
                         total=len(returns_vectors)):
                self.trade(r_vector, S_vector, idx_vector, fake_trading_idx)
