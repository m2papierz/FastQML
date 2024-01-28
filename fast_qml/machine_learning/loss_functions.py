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
from typing import Callable

import fast_qml
import numpy as np
from fast_qml import QubitDevice


class LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def _numpy_wrapper() -> Callable:
        def decorator(func: Callable) -> Callable:
            if fast_qml.DEVICE == QubitDevice.CPU.value:
                np_module = importlib.import_module('pennylane.numpy')
            elif fast_qml.DEVICE == QubitDevice.CPU_JAX.value:
                np_module = importlib.import_module('jax.numpy')
            else:
                NotImplementedError()

            def wrapper(*args, **kwargs):
                original_numpy = globals()['np']
                globals()['np'] = np_module

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    globals()['np'] = original_numpy
                    result = func(*args, **kwargs)
                    print(f"Cannot wrap given function with {np_module}:", e)
                finally:
                    # Restore the original numpy after the function call
                    globals()['np'] = original_numpy

                return result
            return wrapper
        return decorator

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
        @self._numpy_wrapper()
        def _wrapped_loss():
            return self._loss_fn(
                y_real=y_real, y_pred=y_pred
            )
        return _wrapped_loss()


class MSELoss(LossFunction):
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        loss = np.sum((y_real - y_pred) ** 2) / len(y_real)
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
        huber_loss = np.where(
            np.abs(error) < self.delta, 0.5 * error ** 2,
            self.delta * (np.abs(error) - 0.5 * self.delta)
        )
        loss = np.mean(huber_loss)
        return loss


class LogCoshLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def _loss_fn(
            self,
            y_real: np.ndarray,
            y_pred: np.ndarray
    ) -> float:
        log_cosh_loss = np.log(np.cosh(y_real - y_pred))
        loss = np.mean(log_cosh_loss)
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
        y_pred = np.clip(y_pred, self._eps, 1 - self._eps)
        loss = -np.mean(
            y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)
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
        y_pred = np.clip(y_pred, self._eps, 1 - self._eps)
        loss = -np.sum(y_real * np.log(y_pred))
        loss /= len(y_real)
        return loss
