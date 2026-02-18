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

    def get_observation(self, oracle, candidates=None):

        # Sample proportionately to acquisition function
        if self.sample:
            assert (
                oracle.bounds.shape[-1] == 1
            ), "Sampling currently only implemented for 1D functions"
            # Create dense grid over bounds
            X_grid = torch.linspace(
                oracle.bounds[0].item(), oracle.bounds[1].item(), 1000, dtype=torch.float64
            ).unsqueeze(
                -1
            )  # (1000, 1)

            # Evaluate acquisition function
            acq_values = self.acq_func(X_grid.reshape(-1, 1, 1))  # (1000, 1, 1)
            acq_values = acq_values.squeeze()  # (1000,)

            # Normalize and sample
            probs = acq_values / acq_values.sum()
            indices = torch.multinomial(
                probs, num_samples=self.sample_batch_size
            )  # (sample_batch_size,)

            # Take top sample from batch
            sampled_acq = acq_values[indices]
            best_idx = sampled_acq.argmax()
            new_X = X_grid[indices[best_idx]].unsqueeze(-1)
            acq_val = sampled_acq[best_idx]

        # Maximize acquisition function
        else:
            if candidates is not None:
                # Discrete setting
                new_X, acq_val = optimize_acqf_discrete(
                    acq_function=self.acq_func, q=1, choices=candidates, X_avoid=self.model.train_X
                )
            else:
                # Continuous setting
                new_X, acq_val = optimize_acqf(
                    acq_function=self.acq_func,
                    bounds=oracle.bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
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

    def __init__(self, sample: bool = False, sample_batch_size: int = 1):
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ExpectedImprovement(model=model.model, best_f=self.best_f)


class LogEIAcquisition(Acquisition):
    """Log expected improvement acquisition function."""

    def __init__(self, sample: bool = False, sample_batch_size: int = 1):
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogExpectedImprovement(model=model.model, best_f=self.best_f)


class PIAcquisition(Acquisition):
    """Probability of improvement acquisition function."""

    def __init__(self, sample: bool = False, sample_batch_size: int = 1):
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = ProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class LogPIAcquisition(Acquisition):
    """Log probability of improvement acquisition function."""

    def __init__(self, sample: bool = False, sample_batch_size: int = 1):
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.best_f = model.train_y.max().item()
        self.acq_func = LogProbabilityOfImprovement(model=model.model, best_f=self.best_f)


class UCBAcquisition(Acquisition):
    """UCB acquisition function."""

    def __init__(
        self,
        beta: float = 1.0,
        sample: bool = False,
        sample_batch_size: int = 1,
    ):
        self.beta = beta
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = UpperConfidenceBound(model=model.model, beta=self.beta)


class TSAcquisition(Acquisition):
    """Thompson sampling acquisition function."""

    def __init__(self, sample: bool = False, sample_batch_size: int = 1):
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model
        self.acq_func = PathwiseThompsonSampling(model.model)


class KGAcquisition(Acquisition):
    """Knowledge gradient acquisition function."""

    def __init__(
        self,
        num_fantasies: int = 4,
        sample: bool = False,
        sample_batch_size: int = 1,
    ):
        self.num_fantasies = num_fantasies
        self.sample = sample
        self.sample_batch_size = sample_batch_size
        self.model = None

    def update(self, model: SurrogateModel):
        self.model = model

        with torch.no_grad():
            current_value = model.model.posterior(model.model.train_inputs[0]).mean.max()

        self.acq_func = qKnowledgeGradient(
            model=model.model, num_fantasies=self.num_fantasies, current_value=current_value
        )
