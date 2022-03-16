import numpy as np
import torch
from tqdm import tqdm
import cvxpy as cvx
import matplotlib.pyplot as plt

from trader import Trader
from generic import warn
from torch_generic import is_full_rank
from grad_generic import grad_ud, jacobian_ud, set_grad
import config


class BilevelTrader(Trader):
    def __init__(self, transaction_cost_model, hold_cost_model, risk_model,
                 hp, gamma_risk=5., gamma_trade=1., gamma_hold=1., w0=None, v0=1.):
        super().__init__(transaction_cost_model, hold_cost_model, risk_model, gamma_risk,
                         gamma_trade, gamma_hold, w0, v0)
        # Transfer required arrays to torch
        torch.manual_seed(config.seed)
        if w0 is None:
            self.w0 = None
            self.w = None
        else:
            self.w0 = torch.from_numpy(self.w0)
            self.w = self.w0.clone().requires_grad_()
        # Define additional fields
        self.hp = hp
        self.lambd = None
        self.z_vector = []
        self.dz_dhp_vector = []
        self.jacobians = {}
        self.final_utility = None
        # Don't forget to update reset()!

    def reset(self):
        super().reset()
        self.w.requires_grad_()
        self.z_vector = []
        self.dz_dhp_vector = []

    def get_hp(self, idx):
        return self.hp[0, idx]

    @torch.no_grad()
    def set_initial_portfolio(self, w0):
        self.w0 = torch.from_numpy(w0)
        self.w = self.w0.clone().requires_grad_()

    @torch.no_grad()
    def update_v(self):
        self.v *= 1 + self.R_vector[-1].item()

    def eval_R_normalized(self, z, r):
        w_plus = self.w + z
        trans_cost = self.transaction_cost_model.eval_cost_normalized_torch(z, self.v)
        hold_cost = self.hold_cost_model.eval_cost_normalized_torch(w_plus)
        return (r @ w_plus.T - trans_cost - hold_cost).squeeze()

    @torch.no_grad()
    def eval_objective_normalized(self, z, r, cov, w=None):
        if w is None:
            w = self.w.detach()
        w_plus = w + z
        trans_cost = self.transaction_cost_model.eval_cost_normalized(z, self.v)
        hold_cost = self.hold_cost_model.eval_cost_normalized(w_plus)
        risk = self.risk_model.eval_risk_normalized(w_plus, cov)
        return r @ w_plus.T - self.gamma_trade * trans_cost - \
               self.gamma_hold * hold_cost - self.gamma_risk * risk

    def eval_objective_normalized_torch(self, z, r, cov, w=None):
        if w is None:
            w = self.w
        w_plus = w + z
        trans_cost = self.transaction_cost_model.eval_cost_normalized_torch(z, self.v)
        hold_cost = self.hold_cost_model.eval_cost_normalized_torch(w_plus)
        risk = self.risk_model.eval_risk_normalized_torch(w_plus, cov).squeeze()
        return (r @ w_plus.T
                - self.get_hp(config.GAMMA_TRADE_IDX) * trans_cost
                - self.get_hp(config.GAMMA_HOLD_IDX) * hold_cost
                - self.get_hp(config.GAMMA_RISK_IDX) * risk
                ).squeeze()

    @torch.no_grad()
    def forward(self, returns, covariance, active_set):
        z = cvx.Variable(self.w.shape)
        w_plus = self.w.detach() + z
        objective = self.eval_objective_normalized(z, returns, covariance)
        constraints = [constraint.generate(
            w=self.w.detach(), z=z, w_plus=w_plus,
            idxs=active_set
        ) for constraint in self.constraints]
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
        self.z = torch.from_numpy(z.value).requires_grad_()
        self.z_vector.append(self.z)

    def eval_f_g_backward(self, returns, covariance):
        w = self.w
        z = self.z
        g = torch.hstack(
            [torch.squeeze(
                c.evaluate_torch(z=z, w=w, w_plus=w + z)) for c in self.constraints]
        ).unsqueeze(0)
        is_equality = \
            torch.hstack([torch.squeeze(c.is_equality_vector()) for c in self.constraints])
        E_activeI_idxs = torch.abs(g.squeeze()) < config.inactive_constraint_tol
        # A sanity check
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
        f = self.eval_objective_normalized_torch(z, returns, covariance)
        return f, g

    def eval_jacobians(self, returns, covariance, *args):
        w = self.w
        z = self.z
        f, g = self.eval_f_g_backward(returns, covariance, *args)
        gz = jacobian_ud(g, z, create_graph=True, retain_graph=True)
        if not is_full_rank(gz):
            raise Exception('LICQ not hold!')
        lambd_gz = self.lambd.detach() @ gz
        lambd_gzz = jacobian_ud(lambd_gz, z, retain_graph=True)
        lambd_gzh = jacobian_ud(lambd_gz, self.hp, retain_graph=True)
        lambd_gzw = jacobian_ud(lambd_gz, w, retain_graph=True)
        if g.numel() == 1:
            lambd_gzz.unsqueeze_(0)
            lambd_gzh.unsqueeze_(0)
            lambd_gzw.unsqueeze_(0)
        fz = grad_ud(f, z, create_graph=True, retain_graph=True)
        fzz = jacobian_ud(fz, z, retain_graph=True)
        fzh = jacobian_ud(fz, self.hp, retain_graph=True)
        fzw = jacobian_ud(fz, w)
        gh = jacobian_ud(g, self.hp, retain_graph=True)
        gw = jacobian_ud(g, w)

        # Store Jacobians
        self.jacobians['gz'] = gz
        self.jacobians['lambd_gzz'] = lambd_gzz
        self.jacobians['lambd_gzh'] = lambd_gzh
        self.jacobians['lambd_gzw'] = lambd_gzw
        self.jacobians['fzz'] = fzz
        self.jacobians['fzh'] = fzh
        self.jacobians['fzw'] = fzw
        self.jacobians['gh'] = gh
        self.jacobians['gw'] = gw

    def eval_parz_parhp(self):
        gz = self.jacobians['gz']
        lambd_gzz = self.jacobians['lambd_gzz']
        lambd_gzh = self.jacobians['lambd_gzh']
        fzz = self.jacobians['fzz']
        fzh = self.jacobians['fzh']
        gh = self.jacobians['gh']
        A = gz
        B = fzh + lambd_gzh
        C = gh
        H = fzz + lambd_gzz
        if not is_full_rank(H):
            warn('singular H!')
        Hi = torch.pinverse(H)
        A_Hi = A @ Hi
        M = A_Hi @ A.T
        if not is_full_rank(M):
            warn('singular M!')
        # Performance: Is torch.solve faster than torch.inverse?
        Hi_AT_Mi = Hi @ A.T @ torch.pinverse(M)
        return Hi_AT_Mi @ (A_Hi @ B - C) - Hi @ B, (Hi, A_Hi, Hi_AT_Mi)

    def eval_dz_dw(self, jacobians):
        gw = self.jacobians['gw']
        lambd_gzw = self.jacobians['lambd_gzw']
        fzw = self.jacobians['fzw']
        B = fzw + lambd_gzw
        C = gw
        Hi, A_Hi, Hi_AT_Mi = jacobians
        return Hi_AT_Mi @ (A_Hi @ B - C) - Hi @ B

    def eval_dz_dzminus(self, jacobians):
        dz_dw = self.eval_dz_dw(jacobians)
        dw_dzminus = jacobian_ud(self.w, self.z_vector[-2], retain_graph=True)
        return dz_dw @ dw_dzminus

    # returns dz_dhp
    def backward(self, returns, covariance, *args):
        self.eval_jacobians(returns, covariance, *args)
        parz_parh, jacobians = self.eval_parz_parhp()
        if len(self.dz_dhp_vector) == 0:
            self.dz_dhp_vector.append(parz_parh)
            return parz_parh
        dz_dzminus = self.eval_dz_dzminus(jacobians)
        dz_dh = parz_parh + dz_dzminus @ self.dz_dhp_vector[-1]
        self.dz_dhp_vector.append(dz_dh)
        return dz_dh

    def eval_final_utility(self):
        return torch.prod(torch.stack(self.R_vector) + 1)

    def eval_du_dhp(self):
        # dz_dh should already be evaluated and stored.
        # du_dh = \Sum_{t=0}^{T-1}\frac{du}{dz_t}\frac{dz_t}{dh}
        self.final_utility = self.eval_final_utility()
        du_dz_vector = torch.autograd.grad(self.final_utility, self.z_vector)
        du_dhp = torch.bmm(torch.stack(du_dz_vector), torch.stack(self.dz_dhp_vector)).sum(0)
        return du_dhp

    def trade(self, returns, covariance, active_set):
        returns = returns[0].view(1, -1)
        self.forward(returns, covariance, active_set)
        self.backward(returns, covariance)
        self.propagate(self.z, returns)

    def optimize_hp(self, returns_vector, covariance_vector, active_set_vector, iters=int(1e3),
                    lr=1e-6):
        # Initialize
        utility_vector = []
        optimizer = torch.optim.Adam([self.hp], lr=lr)
        for _ in tqdm(range(iters)):
            # Reset
            self.reset()
            # Trade
            self.trade_vector(returns_vector, covariance_vector, active_set_vector)
            # Update
            h_grad = self.eval_du_dhp()
            # Minus to maximize
            set_grad(self.hp, -h_grad)
            optimizer.step()
            # Track
            with torch.no_grad():
                utility_vector.append(self.v)
            if len(utility_vector) > 0 and len(utility_vector) % 5 == 0:
                plt.plot(utility_vector)
                plt.show()
        plt.plot(utility_vector)
        plt.show()
        print(end='')
