# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from typing import Callable
import numpy as np
import pennylane as qml

from fast_qml.machine_learning.estimator import QuantumEstimator
from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.loss_functions import (
    MSELoss, HuberLoss, LogCoshLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
)


class VariationalQuantumEstimator(QuantumEstimator):
    """
    A quantum estimator using a variational approach.
    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: Callable,
            loss_fn: Callable,
            measurements_num: int
    ):
        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )

    def _initialize_weights(self) -> np.ndarray:
        weights = 0.1 * qml.numpy.random.random(
            self._ansatz.params_num, requires_grad=True)
        return weights

    def _quantum_layer(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ):
        self._feature_map.apply(features=x_data)
        self._ansatz.apply(params=weights)

    def _q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ):
        @qml.qnode(device=self._device, interface=self._interface)
        def _circuit():
            self._quantum_layer(
                weights=weights, x_data=x_data)
            return [
                qml.expval(self._measurement_op(i))
                for i in range(self._measurements_num)
            ]
        return _circuit()

    def fit(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            verbose: bool,
            batch_size: int = None
    ):
        optimizer = self._optimizer(
            params=self._weights,
            q_node=self._q_model,
            loss_fn=self._loss_fn,
            batch_size=batch_size,
            epochs_num=num_epochs,
            learning_rate=learning_rate
        )

        optimizer.optimize(
            data=x_data, targets=y_data, verbose=verbose
        )

        return optimizer.weights


class VQRegressor(VariationalQuantumEstimator):
    """
    A variational quantum estimator for regression tasks.
    """

    _allowed_losses = [MSELoss, HuberLoss, LogCoshLoss]

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            measurement_op: Callable = qml.PauliZ
    ):
        if not any(isinstance(loss_fn, loss) for loss in self._allowed_losses):
            raise AttributeError("Invalid loss function.")

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            measurement_op=measurement_op,
            measurements_num=1
        )


class VQClassifier(VariationalQuantumEstimator):
    """
    A variational quantum estimator for classification tasks.
    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            classes_num: int,
            measurement_op: Callable = qml.PauliZ
    ):
        self._classes_num = classes_num

        if classes_num == 2:
            loss_fn = BinaryCrossEntropyLoss()
            measurements_num = 1
        elif classes_num > 2:
            loss_fn = CrossEntropyLoss()
            measurements_num = classes_num
        else:
            raise ValueError("Classes must be 2 or more.")

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )
