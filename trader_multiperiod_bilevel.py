import numpy as np
import cvxpy as cvx
import torch

from trader_bilevel import BilevelTrader
from generic import warn
import config


class MultiPeriodBilevelTrader(BilevelTrader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.w0 is None:
            self.num_assets = None
        else:
            self.num_assets = self.w0.shape[1]

    @torch.no_grad()
    def set_initial_portfolio(self, w0):
        super().set_initial_portfolio(w0)
        self.num_assets = self.w0.shape[1]

    def get_z_trade(self):
        return self.z[:, :self.num_assets]

    @torch.no_grad()
    def forward(self, returns_vector, covariance, active_set, fake_trading_idx):
        fake_trading_idx = [0] + fake_trading_idx
        var_vector = []
        objective = 0.
        constraints = []
        w = self.w.detach()
        for t, returns in enumerate(returns_vector):
            if t not in fake_trading_idx:
                var = np.zeros(w.shape)
                objective += self.eval_objective_normalized(var, returns, covariance, w=w)
            else:
                var = cvx.Variable(w.shape)
                var_vector.append(var)
                w_plus = w + var
                objective += self.eval_objective_normalized(var, returns, covariance, w=w)
                constraints += [constraint.generate(
                    w=w, z=var, w_plus=w_plus,
                    idxs=active_set
                ) for constraint in self.constraints]
                w = w_plus
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        # No trading if there's no solution
        if prob.status == 'infeasible':
            raise Exception('Problem infeasible.')
        self.lambd = torch.from_numpy(
            np.hstack(
                [np.squeeze(c.dual_variables[0].value) for c in constraints]
            )
        ).unsqueeze(0)
        self.z = torch.tensor([var.value[0] for var in var_vector]).view(1, -1).requires_grad_()
        self.z_vector.append(self.z)

    def eval_f_g_backward(self, returns_vector, covariance, fake_trading_idx):
        w = self.w
        z = self.z
        fake_trading_idx = [0] + fake_trading_idx
        objective = 0.
        constraints = []
        is_equality = []
        w = self.w
        vars = self.z.view(-1, self.num_assets)
        for i, var in enumerate(vars):
            var = var.unsqueeze(0)
            returns = returns_vector[fake_trading_idx[i]]
            w_plus = w + var
            objective += self.eval_objective_normalized_torch(var, returns, covariance, w=w)
            constraints += [torch.squeeze(c.evaluate_torch(
                w=w, z=var, w_plus=w_plus)) for c in self.constraints]
            is_equality += [torch.squeeze(c.is_equality_vector()) for c in self.constraints]
            w = w_plus
        g = torch.hstack([torch.squeeze(c) for c in constraints]).unsqueeze(0)
        is_equality = torch.hstack(is_equality)
        E_activeI_idxs = torch.abs(g.squeeze()) < config.inactive_constraint_tol
        # Debugging: equality with non-zero value
        if torch.any(torch.logical_and(is_equality, ~E_activeI_idxs)):
            raise Exception
        # Find (active) inequality constraints with zero lambd
        if (torch.logical_and(
                ~is_equality,
                torch.logical_and(
                    E_activeI_idxs,
                    torch.abs(self.lambd.squeeze()) < config.inactive_constraint_tol))
        ).any():
            warn('One-sided gradient.')
        g = g[:, E_activeI_idxs]
        self.lambd = self.lambd[:, E_activeI_idxs]
        return objective, g

    def trade(
            self, returns_vector, covariance, active_set, fake_trading_idx=[5, 25]):
        self.forward(returns_vector, covariance, active_set, fake_trading_idx)
        self.backward(returns_vector, covariance, fake_trading_idx)
        self.propagate(self.get_z_trade(), returns_vector[0])

    def trade_vector(self, returns_vectors, covariance_vectors, active_set_vectors,
                     fake_trading_idx=None):
        if fake_trading_idx is None:
            for r_vector, S_vector, idx_vector in \
                    zip(returns_vectors, covariance_vectors, active_set_vectors):
                self.trade(r_vector, S_vector, idx_vector)
        else:
            for r_vector, S_vector, idx_vector in \
                    zip(returns_vectors, covariance_vectors, active_set_vectors):
                self.trade(r_vector, S_vector, idx_vector, fake_trading_idx)
