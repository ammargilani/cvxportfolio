from abc import ABC, abstractmethod
import torch


class Constraint(ABC):
    is_equality = None

    def __init__(self, is_equality, dim=None):
        self.__class__.is_equality = torch.tensor(is_equality)
        self.dim = dim

    def generate(self, **kwargs):
        if self.is_equality:
            return self.evaluate(**kwargs) == 0
        else:
            return self.evaluate(**kwargs) <= 0

    def is_equality_vector(self):
        if self.dim is None:
            raise Exception('Constraint has not been evaluated yet.')
        else:
            return self.is_equality.repeat(self.dim)

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def evaluate_torch(self, **kwargs):
        pass
