from abc import ABC, abstractmethod


class RiskModel(ABC):

    @abstractmethod
    def eval_risk(self, **kwargs):
        pass

    @abstractmethod
    def eval_risk_normalized(self, **kwargs):
        pass
