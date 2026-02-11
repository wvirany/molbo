import time

import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from molbo.acquisition import Acquisition
from molbo.models import SurrogateModel
from molbo.oracle import Oracle


class BOLoop:
    """
    BO loop.

    Args:
        model: Surrogate model
        acquisition: Acquisition
        oracle: TODO
        is_continuous: bool (default True) Whether the input space is continuous or discrete
        candidates: torch.Tensor (default None) B x N x d candidate points for discrete optimization

        metrics: BOMetrics (default None) Handles metrics logging during BO loop
    """

    def __init__(
        self,
        model: SurrogateModel,
        acquisition: Acquisition,
        oracle: Oracle,
        is_continuous: bool = True,
        candidates: torch.Tensor = None,
        metrics=None,
    ):
        self.model = model
        self.acquisition = acquisition
        self.oracle = oracle
        self.is_continuous = is_continuous
        self.candidates = candidates
        self.metrics = metrics

        self.history = {
            "X_init": model.train_X,
            "y_init": model.train_y,
            "X_observed": torch.tensor([], dtype=torch.float64),
            "y_observed": torch.tensor([], dtype=torch.float64),
            "acq_vals": [],
            "iteration": [],
            "time_per_iter": [],
            "model_loss": [],
        }

        if self.metrics is not None:
            self.metrics.initialize(self.history)

    def run(self, n_iters):

        for i in range(n_iters):
            iter_start = time.time()

            self.model.fit()
            acq_func = self.acquisition(model=self.model)
            # new_X, acq_val = acq_func.eval() - refactor to this

            if self.is_continuous:
                new_X, acq_val = optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.oracle.bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )
            else:
                new_X, acq_val = optimize_acqf_discrete(
                    acq_function=acq_func,
                    q=1,
                    choices=self.candidates,
                )

            new_y = self.oracle(new_X).unsqueeze(-1)

            self.model.update(new_X, new_y)

            self.history["time_per_iter"].append(time.time() - iter_start)
            self.history["X_observed"] = torch.cat((self.history["X_observed"], new_X))
            self.history["y_observed"] = torch.cat((self.history["y_observed"], new_y))
            self.history["iteration"].append(i)
            self.history["acq_vals"].append(acq_val)
            self.history["model_loss"].append(self.model.loss().item())

            if self.metrics is not None:
                self.metrics.update(i)

        self.model.fit()

        return self.history
