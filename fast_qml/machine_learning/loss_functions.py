# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import importlib
from abc import abstractmethod

import fast_qml
import numpy as np
from fast_qml import QubitDevice


class LossFunction:
    """
    A base class for implementing loss functions. It provides a framework for defining loss functions
    compatible with different numpy modules based on the quantum device being used. It automatically
    selects the appropriate numpy module depending on the device configuration.

    Attributes:
        _np_module: The numpy module appropriate for the current quantum device.
                    This module is used for all numerical operations within the loss function.

    Raises:
        NotImplementedError: If the class is instantiated with an unsupported quantum device configuration.

    Note:
        To use this class, one should subclass it and implement the `_loss_fn` method with the specific
        loss function logic. The subclass can then be instantiated and used directly.
    """
    def __init__(self):
        self._np_module = self._get_numpy_module()

    @staticmethod
    def _get_numpy_module():
        if fast_qml.DEVICE == QubitDevice.CPU.value:
            return importlib.import_module('pennylane.numpy')
        elif fast_qml.DEVICE == QubitDevice.CPU_JAX.value:
            return importlib.import_module('jax.numpy')
        else:
            NotImplementedError()

    @abstractmethod
    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        pass

    def __call__(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        return self._loss_fn(y_real, y_pred)


class MSELoss(LossFunction):
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        loss = self._np_module.sum((y_real - y_pred) ** 2) / len(y_real)
        return loss


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        error = y_real - y_pred
        huber_loss = self._np_module.where(
            self._np_module.abs(error) < self.delta, 0.5 * error ** 2,
            self.delta * (self._np_module.abs(error) - 0.5 * self.delta)
        )
        loss = self._np_module.mean(huber_loss)
        return loss


class LogCoshLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        log_cosh_loss = self._np_module.log(np.cosh(y_real - y_pred))
        loss = self._np_module.mean(log_cosh_loss)
        return loss


class BinaryCrossEntropyLoss(LossFunction):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self._eps = eps

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        y_pred = self._np_module.clip(y_pred, self._eps, 1 - self._eps)
        loss = -self._np_module.mean(
            y_real * self._np_module.log(y_pred) + (1 - y_real) * self._np_module.log(1 - y_pred)
        )
        return loss


class CrossEntropyLoss(LossFunction):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self._eps = eps

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        y_pred = self._np_module.clip(y_pred, self._eps, 1 - self._eps)
        loss = -self._np_module.sum(y_real * self._np_module.log(y_pred))
        loss /= len(y_real)
        return loss
