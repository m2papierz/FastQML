# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import jax
import pennylane as qml

from jax import numpy as jnp
from abc import abstractmethod

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.optimizer import JITOptimizer

JITOptimizer.register_pytree_node()


class VariationalModel:
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm
    ):
        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz

        self._weights = self._init_weights()
        self._device = qml.device(
            name="default.qubit.jax", wires=n_qubits
        )

    def _init_weights(self) -> jnp.ndarray:
        return jax.random.normal(
            key=jax.random.PRNGKey(42),
            shape=(self._ansatz.get_params_num(),)
        )

    def _q_model(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray
    ):
        @jax.jit
        @qml.qnode(device=self._device, interface='jax')
        def _quantum_circuit():
            self._feature_map.apply(features=x_data)
            self._ansatz.apply(params=weights)
            return qml.expval(qml.PauliZ(0))
        return _quantum_circuit()

    def fit(
            self,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
            epochs: int
    ):
        optimizer = JITOptimizer(
            params=self._weights, q_node=self._q_model)

        self._weights = optimizer.optimize(
            data=x_data,
            targets=y_data,
            epochs=epochs
        )

        return self._weights

    @abstractmethod
    def predict_proba(self):
        pass

    @abstractmethod
    def predict(self):
        pass
