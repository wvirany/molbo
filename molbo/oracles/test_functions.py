"""
A collection of factory functions to generate AnalyticOracle objects for different test functions.
"""

import torch

from molbo.oracles import AnalyticOracle


def gaussian_mixture_1d(noise_std=0.0):
    """Multi-modal 1D test function - mixture of 6 Gaussians."""

    def pdf(x, mean, std):
        return (1 / (std * torch.sqrt(2 * torch.tensor(torch.pi)))) * torch.exp(
            -((x - mean) ** 2) / (2 * std**2)
        )

    def f(X):
        g1 = 0.3 * pdf(X, mean=0.0, std_dev=1.0)
        g2 = 0.25 * pdf(X, mean=2.2, std_dev=0.6)
        g3 = 0.2 * pdf(X, mean=4.2, std_dev=0.7)
        g4 = 0.2 * pdf(X, mean=5.8, std_dev=0.7)
        g5 = 0.5 * pdf(X, mean=8.0, std_dev=1.2)
        g6 = 0.1 * pdf(X, mean=10.1, std_dev=0.6)
        return 10 * (g1 + g2 + g3 + g4 + g5 + g6)

    return AnalyticOracle(f=f, bounds=(0.0, 10.0), noise_std=noise_std, optimal_value=1.791)
