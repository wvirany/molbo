# Key Ingredients

- `Oracle`
- `SurrogateModel`
- `Acquisition`
- `BOLoop`

# Examples

## Optimizing an analytic function in 1D

The `AnalyticOracle` class works well when you know an analytic form for your objective function. There are factory functions for defining oracles already implemented in `test_functions.py`; however, the following demonstrates how to define a new oracle simply by specifying the evaluation method.

```python
from molbo.oracle import AnalyticOracle

def f(X):
	return - 2 * X**2 + 1.5 * X - 3

# BoTorch requires an explicit `d` dimension
bounds = torch.tensor([-3.0, 3.0]).unsqueeze(-1)

oracle = AnalyticOracle(
	f=f,
	bounds=bounds,
	noise_std=1.0,          # Default is 0.0
	optimal_value=-2.71875, # Optional, used for regret calculations
)
```

We can generate a few initial training samples and visualize the functiona as follows:

```python
xs = torch.linspace(-3.0, 3.0, 1000)
y = f(xs)

bounds = oracle.bounds

# Sample 3 points uniformly
n_init = 3
train_X = bounds[0].item() + (torch.rand(n_init, 1, dtype=torch.float64) * bounds.diff(dim=0).item())
train_y = oracle(train_X)

plt.plot(xs, y, c='k', ls='--', zorder=1)
plt.scatter(train_X, train_y, c='k')
```

![[Pasted image 20260216133148.png]]

Now we can run a single iteration of a BO loop:

```python
from molbo.acquisition import LogEIAcquisition
from molbo.bo import BOLoop
from molbo.models import GPModel
from molbo.utils import plot_1d_interactive

model = GPModel(train_X, train_y)
acq_func = LogEIAcquisition(model)
bo_loop = BOLoop(model, acq_func, oracle)
history = bo_loop.run(n_iters=1)

plot_1d_interactive(bo_loop)
```

![[Pasted image 20260216133152.png]]

## Optimizing a fixed candidate pool
