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
This module introduces classes for constructing quantum estimators, including Variational Quantum Estimators
(VQE), and Quantum Neural Networks (QNNs). It defines a base class `QuantumEstimator` for general quantum
estimators, along with specialized subclasses for regression and classification tasks.

The estimators are designed to integrate quantum computing techniques into machine learning models,
leveraging quantum feature maps and variational forms. They are capable of handling different types
of loss functions suitable for various machine learning tasks.
"""

from abc import abstractmethod

from typing import Callable
from typing import Union
from typing import List
from typing import Dict
from typing import Mapping

import jax
import numpy as np
import pennylane as qml
from jax import numpy as jnp

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.feature_maps import AmplitudeEmbedding
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.core.estimator import EstimatorParameters
from fast_qml.core.estimator import Estimator


class QuantumEstimator(Estimator):
    """
    Base class for quantum estimator, providing the essential components to construct
    a quantum circuit for machine learning tasks.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        feature_map: Quantum feature map for encoding input data.
        ansatz: Variational form for the quantum circuit.
        layers_num: Number of layers in the quantum circuit.
        measurement_op: Measurement operator for extracting information from the circuit.
        loss_fn: Loss function for model training.
        measurements_num: Number of wires on which to run measurements.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer_fn: Callable,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            measurements_num: int = 1
    ):
        super().__init__(
            loss_fn=loss_fn, optimizer_fn=optimizer_fn, estimator_type='quantum'
        )

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._layers_num = layers_num
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num

        # Validate measurement operation
        if not self._is_valid_measurement_op(measurement_op):
            raise ValueError(
                "Invalid measurement operation provided."
            )

        self._device = qml.device(
            name="default.qubit.jax", wires=self._n_qubits)

        self.params = EstimatorParameters(
            **self._init_parameters(
                n_ansatz_params=ansatz.params_num,
                layers_n=layers_num
            )
        )

    def _init_parameters(
            self,
            n_ansatz_params: Union[int, List[int]],
            layers_n: int = None
    ):
        if layers_n:
            if isinstance(n_ansatz_params, int):
                shape = (layers_n, n_ansatz_params)
            else:
                shape = (layers_n, *n_ansatz_params)
        else:
            if isinstance(n_ansatz_params, int):
                shape = [n_ansatz_params]
            else:
                shape = [*n_ansatz_params]

        weights = 0.1 * jax.random.normal(self._init_rng, shape=shape)
        return {'q_weights': weights}

    @staticmethod
    def _is_valid_measurement_op(measurement_op):
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

    @abstractmethod
    def _quantum_circuit(
            self,
            x_data: jnp.ndarray,
            q_weights: Union[jnp.ndarray, None] = None
    ) -> None:
        """
        Applies the quantum circuit.

        This method is conditional on the data reuploading flag. If data reuploading is enabled,
        the feature map is applied at every layer.

        Args:
            x_data: Input data to be encoded into the quantum state.
            q_weights: Parameters for the variational form.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def model(
            self,
            x_data: jnp.ndarray,
            q_weights: Union[jnp.ndarray, None] = None,
            c_weights: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            training: Union[bool, None] = None
    ):
        """
        Defines estimator model.

        Args:
            x_data: Input data.
            q_weights: Weights of the quantum model.
            c_weights: Weights of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            training: Specifies whether the model is being used for training or inference.

        Returns:
            Outputs of the estimator model.
        """
        @jax.jit
        @qml.qnode(device=self._device, interface="jax")
        def _circuit():
            self._quantum_circuit(x_data=x_data, q_weights=q_weights)
            return [
                qml.expval(self._measurement_op(i))
                for i in range(self._measurements_num)
            ]
        return _circuit()

    def draw_circuit(self) -> None:
        """
        Draws the quantum circuit of the model. This method is particularly useful for debugging
        and understanding the structure of the quantum circuit.
        """
        if isinstance(self._feature_map, AmplitudeEmbedding):
            aux_input = np.random.randn(2 ** self._n_qubits)
        else:
            aux_input = np.random.randn(self._n_qubits)

        aux_input = np.array([aux_input])
        print(qml.draw(self._quantum_circuit)(aux_input, self.params.q_weights))


