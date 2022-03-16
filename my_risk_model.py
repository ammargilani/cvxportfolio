from risk_model import RiskModel
import cvxpy as cvx
from torch_generic import *


class MyRiskModel(RiskModel):

    def __init__(self):
        super(MyRiskModel, self).__init__()

    def eval_risk(self):
        pass

    def eval_risk_normalized(self, w_plus, covariance):
        return cvx.quad_form(w_plus.T, covariance)

    def eval_risk_normalized_torch(self, w_plus, covariance):
        return quad_form_ud(w_plus.T, covariance)
