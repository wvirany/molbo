from molbo.oracles.base import Oracle


class AnalyticOracle(Oracle):

    def __init__(self, f, bounds, noise_std=0.0, optimal_value=None):
        super().__init__(noise_std)
        self.f = f
        self.bounds = bounds
        self._optimal_value = optimal_value

    def _evaluate(self, X):
        return self.f(X)

    @property
    def optimal_value(self):
        if self._optimal_value is None:
            raise ValueError("Optimal value not set. Either set during init or compute manually.")
        return self._optimal_value
