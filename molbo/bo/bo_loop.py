import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from molbo.models.base import SurrogateModel


class BOLoop:
    """
    BO loop.

    Args:
        model: Surrogate model
        acquisition: Acquisition
        oracle: TODO
        is_continuous: bool (default True) Whether the input space is continuous or discrete
        candidates: torch.Tensor (default None) B x N x d candidate points for discrete optimization
    """

    def __init__(
        self,
        model: SurrogateModel,
        acquisition,
        oracle,
        is_continuous: bool = True,
        candidates: torch.Tensor = None,
    ):
        self.model = model
        self.acquisition = acquisition
        self.oracle = oracle
        self.is_continuous = is_continuous
        self.candidates = candidates

    def run(self, n_iters):
        for _ in range(n_iters):
            self.model.fit()

            acq_func = self.acquisition(model=self.model)

            if self.is_continuous:
                new_X, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.oracle.bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )
            else:
                new_X, _ = optimize_acqf_discrete(
                    acq_function=acq_func,
                    q=1,
                    choices=self.candidates,
                )

            new_y = self.oracle(new_X).unsqueeze(-1)

            self.model.update(new_X, new_y)

        self.model.fit()

        return self.model.train_X, self.model.train_y
