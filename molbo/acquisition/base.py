from abc import ABC, abstractmethod

from botorch.acquisition import LogExpectedImprovement

from molbo.models.base import SurrogateModel


class Acquisition(ABC):
    @abstractmethod
    def __call__(self, X):
        pass


class LogEI(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model
        self.best_f = model.train_y.max().item()

    def __call__(self, X):
        acq_func = LogExpectedImprovement(
            model=self.model,
            best_f=self.best_f,
        )
        return acq_func(X)
