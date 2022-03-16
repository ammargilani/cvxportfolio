from constraints.constraint import Constraint


class MaxPortfolio(Constraint):

    def __init__(self, theta):
        super().__init__(is_equality=False)
        self.theta = theta

    def evaluate(self, w_plus, **kwargs):
        self.dim = w_plus.shape
        return w_plus - self.theta.detach()

    def evaluate_torch(self, w_plus, **kwargs):
        return w_plus - self.theta
