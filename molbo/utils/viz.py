import matplotlib.pyplot as plt
import seaborn as sns
import torch

from molbo.bo import BOLoop

sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_1d(bo_loop: BOLoop):
    """Plot an iteration of BO in 1D. Includes objective, surrogate model, and acquisition function."""
    model = bo_loop.model
    f = bo_loop.oracle
    train_X = model.train_X
    X = torch.linspace(
        f.bounds[0].item(),
        f.bounds[1].item(),
        int((f.bounds[1] - f.bounds[0]) * 20),
        dtype=torch.float64,
    ).unsqueeze(-1)
    mean, std = model(X)

    acq = bo_loop.acquisition(model=model)
    acq_values = acq(X.reshape(-1, 1, 1)).squeeze()
    idx = torch.argmax(acq_values)
    new_X = X[idx]

    assert train_X.shape[-1] == 1, "This function only supports 1D inputs."

    fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(6, 4))

    with torch.no_grad():
        ax = axes[0]

        ax.plot(X, f(X), label="Objective function")
        ax.plot(X, mean, label="Posterior mean")
        ax.fill_between(
            X.squeeze(),
            mean.squeeze() - std,
            mean.squeeze() + std,
            alpha=0.2,
            label="Posterior std",
        )
        ax.scatter(train_X, f(train_X), color="k", s=20, zorder=2, label="Observed data")
        ax.scatter(new_X, f(new_X), color="red", s=20, zorder=2)

        ax.legend(bbox_to_anchor=(1.0, 0.7))

        ax = axes[1]

        ax.plot(X, acq_values, color="limegreen")
        ax.scatter(new_X, acq_values[idx], marker="x", color="red", zorder=2)

        plt.show()
