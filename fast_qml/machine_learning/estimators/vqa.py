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


class VariationalQuantumEstimator(QuantumEstimator):
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: qml.operation.Operation,
            loss_fn: Callable
    ):
        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            measurement_op=measurement_op,
            loss_fn=loss_fn
        )

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
            return qml.expval(self._measurement_op)
        return _circuit()

    def fit(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            verbose: bool
    ):
        optimizer = self._optimizer(
            params=self._weights,
            q_node=self._q_model,
            loss_fn=self._loss_fn,
            learning_rate=learning_rate
        )

        self._weights = optimizer.optimize(
            data=x_data,
            targets=y_data,
            epochs=num_epochs,
            verbose=verbose
        )

        return self._weights


class VQRegressor(VariationalQuantumEstimator):
    def __init__(self):
        super().__init__()


class VQClassifier(VariationalQuantumEstimator):
    def __init__(self):
        super().__init__()
