from constraints.constraint import Constraint


class MinPortfolio(Constraint):

    def __init__(self, theta):
        super().__init__(is_equality=False)
        self.theta = theta

    def evaluate(self, w_plus, **kwargs):
        self.dim = w_plus.shape
        return self.theta.detach() - w_plus

    def evaluate_torch(self, w_plus, **kwargs):
        return self.theta - w_plus
