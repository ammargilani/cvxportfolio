import numpy as np
import cvxpy as cvx
from copy import deepcopy

from constraints.no_trade import NoTrade
from constraints.self_finance import SelfFinance
from generic import warn


class Trader:
    def __init__(self, transaction_cost_model, hold_cost_model, risk_model, gamma_risk=5.,
                 gamma_trade=1., gamma_hold=1., w0=None, v0=1.):
        self.w0 = w0
        self.w = np.copy(w0)
        self.v0 = v0
        self.v = self.v0
        self.transaction_cost_model = transaction_cost_model
        self.hold_cost_model = hold_cost_model
        self.risk_model = risk_model
        self.constraints = [SelfFinance(), NoTrade()]
        self.gamma_risk = gamma_risk
        self.gamma_trade = gamma_trade
        self.gamma_hold = gamma_hold
        self.R_vector = []
        self.z = None

    def reset(self):
        self.w = deepcopy(self.w0)
        self.v = self.v0
        self.R_vector = []

    def set_initial_portfolio(self, w0):
        self.w0 = w0
        self.w = w0.copy()

    def eval_h(self):
        return self.v * self.w

    def update_v(self):
        self.v *= 1 + self.R_vector[-1]

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    # eq. (2.10), p. 15
    def eval_R_normalized(self, z, r):
        w_plus = self.w + z
        trans_cost = self.transaction_cost_model.eval_cost_normalized(z, self.v).value
        hold_cost = self.hold_cost_model.eval_cost_normalized(w_plus).value
        return (r @ w_plus.T - trans_cost - hold_cost).squeeze()

    # eq (2.11), p. 15
    def update_weights(self, z, r):
        self.w = 1 / (1 + self.R_vector[-1]) * (1 + r) * (self.w + z)

    def eval_objective_normalized(self, z, r, cov, w=None):
        if w is None:
            w = self.w
        w_plus = w + z
        trans_cost = self.transaction_cost_model.eval_cost_normalized(z, self.v)
        hold_cost = self.hold_cost_model.eval_cost_normalized(w_plus)
        risk = self.risk_model.eval_risk_normalized(w_plus, cov)
        return r @ w_plus.T - self.gamma_trade * trans_cost - \
               self.gamma_hold * hold_cost - self.gamma_risk * risk

    def propagate(self, z, returns):
        R = self.eval_R_normalized(z, returns)
        self.R_vector.append(R)
        self.update_v()
        self.update_weights(z, returns)

    def forward(self, returns, covariance, active_set):
        z = cvx.Variable(self.w.shape)
        w_plus = self.w + z
        objective = self.eval_objective_normalized(z, returns, covariance)
        constraints = [constraint.generate(
            w=self.w, z=z, w_plus=w_plus,
            idxs=active_set
        ) for constraint in self.constraints]
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        # No trading if there's no solution
        if prob.status == 'infeasible':
            warn('No solution!')
            self.z = np.zeros(z.shape)
        else:
            self.z = z.value

    def trade(self, returns, covariance, active_set):
        returns = returns[0].reshape(1, -1)
        self.forward(returns, covariance, active_set)
        self.propagate(self.z, returns)

    def trade_vector(self, returns_vector, covariance_vector, active_set_vector):
        for returns, covariance, active_set in \
                zip(returns_vector, covariance_vector, active_set_vector):
            self.trade(returns, covariance, active_set)

    def eval_final_utility(self):
        return self.v
