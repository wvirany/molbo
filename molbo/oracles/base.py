from abc import ABC, abstractmethod

import torch


class Oracle(ABC):

    def __init__(self, noise_std=0.0):
        """
        Base class for oracles.

        Args:
            noise_std: (m) tensor containing noise standard deviations for each output dimension

        Note: currently only supports homoskedastic additive Gaussian noise.
        """
        self.noise_std = noise_std

    @abstractmethod
    def _evaluate(self, X):
        """
        Internal evaluation method - implemented in subclass.

        Depends on use-case:
        - analytic functions (continuous domain or discrete grid)
        - fixed candidate sets (fixed dataset of points, implemented as a lookup table)
        - blackbox functions (takes arbitrary input)
        """
        pass

    def __call__(self, X):
        """
        Evaluate oracle f: R^d -> R^m. Supports individual or batch evaluation.

        Args:
            X: (B, N, d) tensor containing points to evaluate

        Returns:
            y: (B, N, m) tensor containing outputs
        """
        y_true = self._evaluate(X)

        if self.noise_std > 0:
            y = y_true + torch.randn_like(y_true) * self.noise_std
        else:
            y = y_true

        return y
