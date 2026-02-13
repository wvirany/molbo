from abc import ABC, abstractmethod

import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from molbo.models.base import SurrogateModel


class Acquisition(ABC):
    """Wrapper for BoTorch acquisition functions."""

    def get_observation(self, oracle, is_continuous: bool = True, candidates=None):
        if is_continuous:
            new_X, acq_val = optimize_acqf(
                acq_function=self.acq_func,
                bounds=oracle.bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )
        else:
            new_X, acq_val = optimize_acqf_discrete(
                acq_function=self.acq_func,
                bounds=oracle.bounds,
                q=1,
                choices=candidates,
            )

        return new_X, acq_val

    @abstractmethod
    def update(self, model: SurrogateModel):
        """Updaate acquisition function with new surrogate model."""
        pass

    def __call__(self, X):
        return self.acq_func(X)


class EIAcquisition(Acquisition):
    """Expected improvement acquisition function."""

    def __init__(self, model: SurrogateModel):
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ExpectedImprovement(model=model.model, best_f=self.best_f)


class LogEIAcquisition(Acquisition):
    """Log expected improvement acquisition function."""

    def __init__(self, model: SurrogateModel):
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogExpectedImprovement(model=model.model, best_f=self.best_f)


class PIAcquisition(Acquisition):
    """Probability of improvement acquisition function."""

    def __init__(self, model: SurrogateModel):
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class LogPIAcquisition(Acquisition):
    """Log probability of improvement acquisition function."""

    def __init__(self, model: SurrogateModel):
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class UCBAcquisition(Acquisition):
    """UCB acquisition function."""

    def __init__(self, model: SurrogateModel, beta: float = 1.0):
        self.beta = beta
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = UpperConfidenceBound(model=model.model, beta=self.beta)


class TSAcquisition(Acquisition):
    """Thompson sampling acquisition function."""

    def __init__(self, model: SurrogateModel):
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = PathwiseThompsonSampling(model.model)


class KGAcquisition(Acquisition):
    """Knowledge gradient acquisition function."""

    def __init__(self, model: SurrogateModel, num_fantasies: int = 4):
        self.num_fantasies = num_fantasies
        self.update(model)

    def update(self, model: SurrogateModel):
        self.model = model

        with torch.no_grad():
            current_value = model.model.posterior(model.model.train_inputs[0]).mean.max()

        self.acq_func = qKnowledgeGradient(
            model=model.model, num_fantasies=self.num_fantasies, current_value=current_value
        )
