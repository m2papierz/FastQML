# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

"""
Loss Functions Module.

This module provides implementations of various loss functions used in machine learning. The classes in this
module are intended to be used as loss functions in training loops of machine learning models. An instance of
any of these classes can be created and passed to optimization algorithms to compute the loss between
predicted and actual values.

Example:
    >>> loss_fn = MSELoss()
    >>> loss = loss_fn(y_real, y_pred)
    >>> print("MSE Loss:", loss)

    >>> huber_loss_fn = HuberLoss(delta=1.0)
    >>> loss = huber_loss_fn(y_real, y_pred)
    >>> print("Huber Loss:", loss)

    >>> logcosh_loss_fn = LogCoshLoss()
    >>> loss = logcosh_loss_fn(y_real, y_pred)
    >>> print("LogCosh Loss:", loss)
"""

from abc import abstractmethod
from jax import numpy as jnp


class LossFunction:
    """
    A base class for implementing loss functions. It provides a framework for defining loss functions
    compatible with different numpy modules based on the quantum device being used. It automatically
    selects the appropriate numpy module depending on the device configuration.


    Raises:
        NotImplementedError: If the class is instantiated with an unsupported quantum device configuration.

    Note:
        To use this class, one should subclass it and implement the `_loss_fn` method with the specific
        loss function logic. The subclass can then be instantiated and used directly.
    """
    @abstractmethod
    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> float:
        """
        Abstract method for computing the loss. This method must be implemented by subclasses.

        The method takes in the true values (y_real) and the predicted values (y_pred) and
        computes the loss between them. The specific computation depends on the loss function
        implemented by the subclass.

        Parameters:
            y_real: The true values.
            y_pred: The predicted values.

        Returns:
            The computed loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> float:
        """
        Enables the LossFunction object to be called as a function. This method provides a convenient
        way to compute the loss using the implemented `_loss_fn` method.

        Args:
            y_real: The true values.
            y_pred: The predicted values.

        Returns:
            The computed loss based on the logic defined in the `_loss_fn` method.
        """
        return self._loss_fn(y_real, y_pred)


class MSELoss(LossFunction):
    """
    This class implements the Mean Squared Error (MSE) loss function. The MSE loss calculates
    the average of the squares of the differences between the predicted values (y_pred) and the
    actual values (y_real). This loss function is widely used in regression problems.
    """
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the Mean Squared Error loss.

        Args:
            y_real: The actual values.
            y_pred: The predicted values.

        Returns:
            The computed MSE loss.
        """
        return jnp.mean((y_real - y_pred) ** 2)


class HuberLoss(LossFunction):
    """
    This class implements the Huber loss function, which is less sensitive to outliers in data
    than the squared error loss. The Huber loss is quadratic for small values of error and linear
    for large values, with the transition point set by the 'delta' parameter.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the Huber loss.

        Args:
            y_real: The actual values.
            y_pred: The predicted values.

        Returns:
            float: The computed Huber loss.
        """
        error = y_real - y_pred
        huber_loss = jnp.where(
            jnp.abs(error) < self.delta, 0.5 * error ** 2,
            self.delta * (jnp.abs(error) - 0.5 * self.delta)
        )
        loss = jnp.mean(huber_loss)
        return loss


class LogCoshLoss(LossFunction):
    """
    This class implements the Log-Cosh loss function, which is a smooth version of the L2 norm that
    is less influenced by occasional wildly incorrect predictions.

    Log-Cosh is the logarithm of the hyperbolic cosine of the prediction error and is mainly used
    in regression problems.
    """
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the Log-Cosh loss.

        Args:
            y_real: The actual values.
            y_pred: The predicted values.

        Returns:
            The computed Log-Cosh loss.
        """
        log_cosh_loss = jnp.log(jnp.cosh(y_real - y_pred))
        loss = jnp.mean(log_cosh_loss)
        return loss


class BinaryCrossEntropyLoss(LossFunction):
    """
    BinaryCrossEntropyLoss implements a binary cross-entropy loss function for binary
    classification tasks.

    IT is specifically tailored for scenarios where the output can be classified into
    one of two possible classes (binary classification). It calculates the loss between
    the true binary labels and the predicted probabilities.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self._eps = eps

    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> float:
        y_pred = jnp.clip(y_pred, self._eps, 1 - self._eps)
        loss = -jnp.mean(
            y_real * jnp.log(y_pred) + (1 - y_real) * jnp.log(1 - y_pred)
        )
        return loss


class CrossEntropyLoss(LossFunction):
    """
    CrossEntropyLoss implements a loss function for classification tasks, specifically using the
    cross-entropy loss metric. This class is intended for multi-class classification problems.

    It supports both one-hot encoded targets and class index targets.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self._eps = eps

    @staticmethod
    def _one_hot_encode(y_real, k):
        """
        Creates a one-hot encoding of x of size k
        """
        return jnp.array(y_real[:, None] == jnp.arange(k))

    @staticmethod
    def _is_one_hot(y_real):
        """
        Checks if y is one-hot encoded
        """
        return (
                y_real.ndim == 2 and
                jnp.all(jnp.sum(y_real, axis=1) == 1) and
                jnp.all((y_real == 0) | (y_real == 1))
        )

    def _loss_fn(
            self,
            y_real: jnp.ndarray,
            y_pred: jnp.ndarray
    ) -> float:
        if y_pred.shape[0] != len(y_real):
            y_pred = y_pred.transpose()

        if not self._is_one_hot(y_real):
            y_real = self._one_hot_encode(y_real, y_pred.shape[-1])

        y_pred = jnp.clip(y_pred, self._eps, 1 - self._eps)
        loss = -jnp.sum(y_real * jnp.log(y_pred))
        loss /= len(y_real)
        return loss
