from transaction_cost_model import TransactionCostModel
import cvxpy as cvx
import torch


class MyTransactionCostModel(TransactionCostModel):

    def __init__(self, half_spread, b=1., c=0.):
        super(MyTransactionCostModel, self).__init__()
        self.half_spread = half_spread
        self.b = b
        self.c = c

    def eval_cost(self, u, v, V=1., sigma=0.):
        """
        :param v: the total portfolio value
        :param V: the total market volume traded for the asset in the time period, expressed in dollar value
        :param sigma: std of last 24 hours price
        """
        return cvx.sum(self.half_spread * cvx.abs(u[0, :-1]) + self.b * sigma * cvx.abs(u[0, :-1]) ** 1.5 / (V / v) ** .5 +
                       self.c * u[0, :-1])

    def eval_cost_normalized(self, z, v, V=1., sigma=0.):
        """
        :param v: the total portfolio value
        :param V: the total market volume traded for the asset in the time period, expressed in dollar value
        :param sigma: std of last 24 hours price
        """
        return cvx.sum(self.half_spread * cvx.abs(z[0, :-1]) + self.b * sigma * cvx.abs(z[0, :-1]) ** 1.5 / (V / v) ** .5 +
                       self.c * z[0, :-1])

    def eval_cost_normalized_torch(self, z, v, V=1., sigma=0.):
        """
        :param v: the total portfolio value
        :param V: the total market volume traded for the asset in the time period, expressed in dollar value
        :param sigma: std of last 24 hours price
        """
        return torch.sum(self.half_spread * torch.abs(z[0, :-1]) + self.b * sigma * torch.abs(z[0, :-1]) ** 1.5 / (V / v) ** .5 +
                         self.c * z[0, :-1])
