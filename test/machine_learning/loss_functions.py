import pytest
import numpy as np

from fast_qml.machine_learning.loss_functions import (
    MSELoss, HuberLoss, LogCoshLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
)


# Fixtures for test data
@pytest.fixture
def mse_data():
    return np.array([1, 2, 3]), np.array([1, 2, 5])  # y_real, y_pred


@pytest.fixture
def huber_data():
    return np.array([1, 2, 3, 4]), np.array([2, 2, 3, 5]), 1.0  # y_real, y_pred, delta


@pytest.fixture
def log_cosh_data():
    return np.array([0, 1, 2]), np.array([0, 2, 2])  # y_real, y_pred


@pytest.fixture
def bce_data():
    return np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2])  # y_real, y_pred


@pytest.fixture
def ce_data():
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])  # y_real, y_pred


# Test functions
def test_mse_loss(mse_data):
    y_real, y_pred = mse_data
    mse_loss = MSELoss()
    expected_loss = np.sum((y_real - y_pred) ** 2) / len(y_real)
    calculated_loss = mse_loss(y_real, y_pred)
    assert np.isclose(calculated_loss, expected_loss)


def test_huber_loss(huber_data):
    y_real, y_pred, delta = huber_data
    huber_loss = HuberLoss(delta)
    error = y_real - y_pred
    expected_loss = np.mean(np.where(
        np.abs(error) < delta, 0.5 * error ** 2,
        delta * (np.abs(error) - 0.5 * delta)
    ))
    calculated_loss = huber_loss(y_real, y_pred)
    assert np.isclose(calculated_loss, expected_loss)


def test_log_cosh_loss(log_cosh_data):
    y_real, y_pred = log_cosh_data
    log_cosh_loss = LogCoshLoss()
    expected_loss = np.mean(np.log(np.cosh(y_pred - y_real)))
    calculated_loss = log_cosh_loss(y_real, y_pred)
    assert np.isclose(calculated_loss, expected_loss)


def test_bce_loss(bce_data):
    y_real, y_pred = bce_data
    eps = 1e-12
    bce_loss = BinaryCrossEntropyLoss(eps)
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    expected_loss = -np.mean(y_real * np.log(y_pred_clipped) + (1 - y_real) * np.log(1 - y_pred_clipped))
    calculated_loss = bce_loss(y_real, y_pred)
    assert np.isclose(calculated_loss, expected_loss)


def test_ce_loss(ce_data):
    y_real, y_pred = ce_data
    eps = 1e-12
    ce_loss = CrossEntropyLoss(eps)
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    expected_loss = -np.sum(y_real * np.log(y_pred_clipped)) / len(y_real)
    calculated_loss = ce_loss(y_real, y_pred)
    assert np.isclose(calculated_loss, expected_loss)
