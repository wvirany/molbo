import torch
from botorch.test_functions import SyntheticTestFunction


class Toy1DFunction(SyntheticTestFunction):
    """
    Sub-module of `SyntheticTestFunction`
    """

    def __init__(self, noise_std=None, negate=False, dtype=torch.double):
        """
        noise_std: Add Normal random variable w/ variance noise_std**2
        """
        self.dim = 1
        self._bounds = [(0.0, 10.0)]
        self.continuous_inds = [0]
        self.max = 1.7910193051233148
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X):
        """
        Args:
            X: B x d tensor

        Returns:
            B x m tensor
        """
        pdf = lambda x, mean, std_dev: (
            1 / (std_dev * torch.sqrt(2 * torch.tensor(torch.pi)))
        ) * torch.exp(-((x - mean) ** 2) / (2 * std_dev**2))

        g1 = lambda x: 0.3 * pdf(X, mean=torch.tensor(0), std_dev=torch.tensor(1))
        g2 = lambda x: 0.25 * pdf(X, mean=torch.tensor(2.2), std_dev=torch.tensor(0.6))
        g3 = lambda x: 0.2 * pdf(x, mean=torch.tensor(4.2), std_dev=torch.tensor(0.7))
        g4 = lambda x: 0.2 * pdf(x, mean=torch.tensor(5.8), std_dev=torch.tensor(0.7))
        g5 = lambda x: 0.5 * pdf(x, mean=torch.tensor(8), std_dev=torch.tensor(1.2))
        g6 = lambda x: 0.1 * pdf(x, mean=torch.tensor(10.1), std_dev=torch.tensor(0.6))

        return (10 * (g1(X) + g2(X) + g3(X) + g4(X) + g5(X) + g6(X))).squeeze(-1)
