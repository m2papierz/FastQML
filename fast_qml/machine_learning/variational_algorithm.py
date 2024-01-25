# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from abc import abstractmethod

import fast_qml
import numpy as np
import pennylane as qml

from pennylane import numpy as qnp

from fast_qml import QubitDevice
from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.optimizer import DefaultOptimizer, JITOptimizer


class EstimatorBase:
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: qml.operation.Operation
    ):
        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._measurement_op = measurement_op

        if fast_qml.DEVICE == QubitDevice.CPU.value:
            self._interface = 'auto'
            self._optimizer = DefaultOptimizer
            self._device = qml.device(
                name="default.qubit", wires=n_qubits
            )
        elif fast_qml.DEVICE == QubitDevice.CPU_JAX.value:
            self._interface = 'jax'
            self._optimizer = JITOptimizer
            self._optimizer.register_pytree_node()
            self._device = qml.device(
                name="default.qubit.jax", wires=n_qubits
            )
        else:
            raise NotImplementedError()

        self._weights = self._initialize_weights()

    def _initialize_weights(self) -> np.ndarray:
        weights = 0.1 * qnp.random.random(
            self._ansatz.get_params_num(), requires_grad=True)
        return weights

    def _q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ):
        @qml.qnode(device=self._device, interface=self._interface)
        def _quantum_circuit():
            self._feature_map.apply(features=x_data)
            self._ansatz.apply(params=weights)
            return qml.expval(self._measurement_op)
        return _quantum_circuit()

    @abstractmethod
    def fit(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            verbose: bool
    ):
        pass


class SimpleQuantumEstimator(EstimatorBase):
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: qml.operation.Operation
    ):
        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            measurement_op=measurement_op
        )

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
            learning_rate=learning_rate
        )

        self._weights = optimizer.optimize(
            data=x_data,
            targets=y_data,
            epochs=num_epochs,
            verbose=verbose
        )

        return self._weights
