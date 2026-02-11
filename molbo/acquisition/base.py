from abc import ABC, abstractmethod

from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

from molbo.models.base import SurrogateModel


class Acquisition(ABC):
    @abstractmethod
    def __call__(self, X):
        pass

    # @abstractmethod
    # def optimize():
    #     pass

    # @abstractmethod
    # def sample():
    #     pass


class EI(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model
        self.best_f = model.train_y.max().item()

    def __call__(self, X):
        acq_func = ExpectedImprovement(model=self.model, best_f=self.best_f)
        return acq_func(X)


class LogEI(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model
        self.best_f = model.train_y.max().item()

    def __call__(self, X):
        acq_func = LogExpectedImprovement(model=self.model, best_f=self.best_f)
        return acq_func(X)


class PI(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model
        self.best_f = model.train_y.max().item()

    def __call__(self, X):
        acq_func = ProbabilityOfImprovement(model=self.model, best_f=self.best_f)
        return acq_func(X)


class LogPI(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model
        self.best_f = model.train_y.max().item()

    def __call__(self, X):
        acq_func = LogProbabilityOfImprovement(model=self.model, best_f=self.best_f)
        return acq_func(X)


class UCB(Acquisition):
    def __init__(self, model: SurrogateModel, beta: float = 0.1):
        self.model = model.model
        self.beta = beta

    def __call__(self, X):
        acq_func = UpperConfidenceBound(model=self.model, beta=self.beta)
        return acq_func(X)


class ThompsonSampling(Acquisition):
    def __init__(self, model: SurrogateModel):
        self.model = model.model

    def __call__(self, X):
        acq_func = PathwiseThompsonSampling(model=self.model)
        return acq_func(X)
