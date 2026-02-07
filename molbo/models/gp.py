import gpytorch
import torch
from gauche.kernels.fingerprint_kernels.minmax_kernel import MinMaxKernel

from molbo.models.base import SurrogateModel


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model with MinMax kernel for Morgan fingerprints."""

    def __init__(self, train_x, train_y):
        super().__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel=MinMaxKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSurrogate(SurrogateModel):
    """GP surrogate model for BO."""

    def __init__(
        self,
        training_iter: int = 100,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        self.training_iter = training_iter
        self.learning_rate = learning_rate
        self.device = device

        self.model = None
        self.likelihood = None
        self.train_x = None
        self.train_y = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Fit the GP model.

        Minimizes the negative MLL of the GP model.

        Args:
            X_train: Tensor containing molecular embeddings of shape (n_samples, n_features)
            y_train: Target values of shape (n_samples,)
        """

        self.X_train = X_train.to(self.device, dtype=torch.float64)
        self.y_train = y_train.to(self.device, dtype=torch.float64)

        # Initialize GP model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            self.device, dtype=torch.float64
        )
        self.model = ExactGPModel(self.X_train, self.y_train, self.likelihood).to(
            self.device, dtype=torch.float64
        )

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train the model
        self.model.train()
        self.likelihood.train()

        for i in range(1, self.training_iter + 1):
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -self.mll(output, self.y_train)

            if i % 10 == 0:
                print(
                    f"GP training iteration{i}/{self.training_iter} | Loss: {loss.item():.4f} - Outputscale: {self.gp_model.covar_module.outputscale.item():.4f} - Mean: {self.gp_model.mean_module.constant.item():.4f} - Noise: {self.likelihood.noise.item():.4f}"
                )

            loss.backward()
            self.optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def predict(self, X_test: torch.Tensor):
        """
        Predict the target values for the test data.

        Args:
            X_test: Test features (n_test, n_features)
        """
        X_test = X_test.to(self.device, dtype=torch.float64)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            mean = pred.mean
            std = pred.stddev

        return mean, std

    def update(self, X_new: torch.Tensor, y_new: torch.Tensor):
        """
        Update the GP model with new data.

        Args:
            X_new: New features (n_new, n_features)
            y_new: New target values (n_new,)
        """
        X_new = X_new.to(self.device, dtype=torch.float64)
        y_new = y_new.to(self.device, dtype=torch.float64)

        self.X_train = torch.cat([self.X_train, X_new], dim=0)
        self.y_train = torch.cat([self.y_train, y_new], dim=0)

        self.fit(self.X_train, self.y_train)

        return
