from abc import ABC, abstractmethod


class HoldCostModel(ABC):

    @abstractmethod
    def eval_cost(self, **kwargs):
        pass

    @abstractmethod
    def eval_cost_normalized(self, **kwargs):
        pass
