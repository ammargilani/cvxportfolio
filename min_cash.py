from constraints.constraint import Constraint


class MinCash(Constraint):

    def __init__(self, theta):
        super().__init__(is_equality=False, dim=1)
        self.theta = theta

    def evaluate(self, w_plus, **kwargs):
        return self.theta - w_plus[0, -1]

    def evaluate_torch(self, w_plus, **kwargs):
        return self.evaluate(w_plus)
