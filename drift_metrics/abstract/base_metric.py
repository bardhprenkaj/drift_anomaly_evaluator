from abc import ABC, abstractmethod

class AbstractMetric(ABC):

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    @abstractmethod
    def precision(self):
        pass

    @abstractmethod
    def recall(self):
        pass

    def f_score(self, beta=1):
        p, _ = self.precision()
        r, _ = self.recall()

        return (1 + beta**2) * (p * r) / ((beta**2 * p) + r) if (p != 0 and r != 0) else 0