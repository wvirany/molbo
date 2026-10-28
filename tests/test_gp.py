"""Lightweight tests for GPSurrogate and TanimotoGPSurrogate.

Checks initialize -> fit -> predict works end-to-end,
the featurizer works (TanimotoGPSurrogate accepts SMILES), and the fit warning works.
"""

import pytest
import torch

from molbo.surrogates.featurizer import IdentityFeaturizer, MorganFingerprintFeaturizer
from molbo.surrogates.gp import GPSurrogate, TanimotoGPSurrogate

# --- Fixtures -------------------------------------------------------------

# Continuous data: feature tensors in, feature tensors out.
CONT_X = torch.rand(8, 3, dtype=torch.float64)
CONT_Y = torch.rand(8, 1, dtype=torch.float64)
CONT_X_NEW = torch.rand(4, 3, dtype=torch.float64)
CONT_Y_NEW = torch.rand(4, 1, dtype=torch.float64)

# Molecular data: SMILES in, the surrogate model featurizes internally.
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccncc1"]
SMILES_NEW = ["CCCl", "CCBr"]
MOL_Y = torch.rand(len(SMILES), 1, dtype=torch.float64)
MOL_Y_NEW = torch.rand(len(SMILES_NEW), 1, dtype=torch.float64)


# --- GPSurrogate --------------------------------------------------------------


def test_gp_surrogate_default_featurizer():
    surrogate = GPSurrogate()
    assert isinstance(surrogate.featurizer, IdentityFeaturizer)


def test_gp_surrogate_lifecycle():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()

    mean, std = surrogate(CONT_X)
    assert mean.shape == (8, 1)
    assert std.shape == (8, 1)
    assert torch.all(std > 0)


def test_gp_surrogate_update_grows_data():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()
    surrogate.update(CONT_X_NEW, CONT_Y_NEW)

    assert surrogate.train_X.shape[0] == 12
    assert surrogate.train_y.shape[0] == 12

    # fit() after update should still work
    surrogate.fit()


def test_gp_surrogate_loss_runs():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()
    loss = surrogate.loss()
    assert torch.is_tensor(loss)


# --- TanimotoGPSurrogate ------------------------------------------------------


def test_tanimoto_default_featurizer():
    surrogate = TanimotoGPSurrogate(featurizer=MorganFingerprintFeaturizer())
    assert isinstance(surrogate.featurizer, MorganFingerprintFeaturizer)


def test_tanimoto_accepts_smiles():
    """The surrogate model takes identity-space SMILES and featurizes internally."""
    surrogate = TanimotoGPSurrogate(featurizer=MorganFingerprintFeaturizer())
    surrogate.initialize(SMILES, MOL_Y)
    surrogate.fit()

    # train_X is stored as feature tensors, not SMILES
    assert torch.is_tensor(surrogate.train_X)
    assert surrogate.train_X.shape[0] == len(SMILES)

    mean, std = surrogate(SMILES)
    assert mean.shape == (len(SMILES), 1)
    assert std.shape == (len(SMILES), 1)


def test_tanimoto_update_with_smiles():
    surrogate = TanimotoGPSurrogate(featurizer=MorganFingerprintFeaturizer())
    surrogate.initialize(SMILES, MOL_Y)
    surrogate.fit()
    surrogate.update(SMILES_NEW, MOL_Y_NEW)

    assert surrogate.train_X.shape[0] == len(SMILES) + len(SMILES_NEW)
    assert surrogate.train_y.shape[0] == len(SMILES) + len(SMILES_NEW)
    surrogate.fit()


# --- fit guard ------------------------------------------------------------


def test_predict_before_fit_raises():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        surrogate(CONT_X)


def test_loss_before_fit_raises():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        surrogate.loss()


def test_gp_surrogate_normalize_default_on():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()
    assert surrogate._input_transform is not None  # Normalize on by default


def test_gp_surrogate_normalize_off():
    surrogate = GPSurrogate(normalize=False)
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()
    assert surrogate._input_transform is None  # no input transform
    mean, std = surrogate(CONT_X)
    assert mean.shape == (8, 1) and torch.all(std > 0)


def test_gp_surrogate_parameters_returns_dict():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    surrogate.fit()
    params = surrogate.parameters
    assert isinstance(params, dict) and len(params) > 0


def test_parameters_before_fit_raises():
    surrogate = GPSurrogate()
    surrogate.initialize(CONT_X, CONT_Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        surrogate.parameters
