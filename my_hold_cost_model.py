from hold_cost_model import HoldCostModel
import cvxpy as cvx
import torch
from torch_generic import *


class MyHoldCostModel(HoldCostModel):

    def __init__(self, borrowing_fee):
        super(MyHoldCostModel, self).__init__()
        self.borrowing_fee = borrowing_fee

    def eval_cost(self, h_plus):
        return cvx.sum(self.borrowing_fee * cvx.neg(h_plus[0, :-1]))

    def eval_cost_normalized(self, w_plus):
        return cvx.sum(self.borrowing_fee * cvx.neg(w_plus[0, :-1]))

    def eval_cost_normalized_torch(self, w_plus):
        return torch.sum(self.borrowing_fee * neg_ud(w_plus[0, :-1]))
