import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_1d(model, X, train_X, new_X, f, acq_func):

    pred = model.posterior(X.unsqueeze(-1))
    mean, std = pred.mean.squeeze(), pred.stddev.squeeze()

    fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(6, 4))

    with torch.no_grad():
        ax = axes[0]

        ax.plot(X, f(X), label="Objective function")
        ax.plot(X, mean, label="Posterior mean")
        ax.fill_between(X.squeeze(), mean - std, mean + std, alpha=0.2, label="Posterior std")
        ax.scatter(train_X, f(train_X), color="k", s=20, zorder=2, label="Observed data")
        ax.scatter(new_X, f(new_X), color="red", s=20, zorder=2)

        ax.legend(bbox_to_anchor=(1.0, 0.7))

        ax = axes[1]

        ax.plot(X, acq_func(X.reshape(-1, 1, 1)), color="limegreen")
        ax.scatter(new_X, acq_func(new_X.unsqueeze(-1)), marker="x", color="red", zorder=2)

        plt.show()
