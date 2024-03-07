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
This module provides classes for building variational quantum estimators . It includes a general
`VariationalQuantumEstimator` class, which serves as a base for specialized estimators like `VQRegressor`
for regression tasks and `VQClassifier` for classification tasks.

The estimators are designed to integrate quantum computing techniques into machine learning models,
leveraging quantum feature maps and variational forms. They are capable of handling different types
of loss functions suitable for various machine learning tasks.
"""

from typing import Callable
from typing import Union
from typing import List
from typing import Dict
from typing import Mapping

import jax
import numpy as np
import pennylane as qml
from jax import numpy as jnp

from fast_qml.core.estimator import EstimatorParameters
from fast_qml.core.estimator import Estimator
from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.feature_maps import AmplitudeEmbedding
from fast_qml.quantum_circuits.variational_forms import VariationalForm


class VariationalQuantumEstimator(Estimator):
    """
    A quantum estimator using a variational approach. This class provides the foundation for building
    quantum machine learning models with variational circuits. It initializes the quantum device,
    defines the quantum circuit with feature maps and variational forms, and implements the training
    algorithm.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        feature_map: Quantum feature map to encode classical data into quantum states.
        ansatz: Variational form representing the parameterized quantum circuit.
        measurement_op: Measurement operator to extract information from the quantum state.
        loss_fn: Loss function for the training process.
        measurements_num: Number of wires on which to run measurements.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: Callable,
            loss_fn: Callable,
            optimizer_fn: Callable,
            measurements_num: int
    ):
        super().__init__(
            loss_fn=loss_fn, optimizer_fn=optimizer_fn, estimator_type='quantum'
        )

        # Validate measurement operation
        if not self._is_valid_measurement_op(measurement_op):
            raise ValueError("Invalid measurement operation provided.")

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num

        self._device = qml.device(
            name="default.qubit.jax", wires=self._n_qubits)

        self.params = EstimatorParameters(
            **self._init_parameters(
                n_ansatz_params=ansatz.params_num
            )
        )

    def _init_parameters(
            self,
            n_ansatz_params: Union[int, List[int]]
    ):
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


    def _quantum_circuit(
            self,
            x_data: jnp.ndarray,
            q_weights: Union[jnp.ndarray, None] = None,
    ) -> None:
        """
        Applies the quantum circuit consisting of the feature map and the variational form.

        This method encodes classical data into quantum states using the specified feature map and
        then applies the variational form with the given weights.

        Args:
            x_data: Classical data to be encoded into quantum states.
            q_weights: Parameters for the variational form.
        """
        self._feature_map.apply(features=x_data)
        self._ansatz.apply(params=q_weights)

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
        @qml.qnode(device=self._device, interface='jax')
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
        print(qml.draw(self._quantum_circuit)(aux_input, self.params.q_weights, ))


class VQRegressor(VariationalQuantumEstimator):
    """
    A variational quantum estimator specifically designed for regression tasks. It inherits from
    VariationalQuantumEstimator and ensures that the loss function used is appropriate for regression.

    This class is suitable for tasks where the goal is to predict continuous or quantitative outputs.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer_fn: Callable,
            measurement_op: Callable = qml.PauliZ
    ):
        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            measurement_op=measurement_op,
            measurements_num=1
        )

    def predict(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the predictions (model outputs) for the given input data.

        Args:
            x: An array of input data.
        """
        return jnp.array(
            self.model(q_weights=self.params.q_weights, x_data=x)
        ).ravel()


class VQClassifier(VariationalQuantumEstimator):
    """
    A variational quantum estimator tailored for classification tasks. Depending on the number of
    classes, it automatically selects a suitable loss function (BinaryCrossEntropyLoss for binary
    classification or CrossEntropyLoss for multi-class classification).

    This class is ideal for tasks where the goal is to categorize inputs into discrete classes.

    Args:
        classes_num: Number of classes in the classification problem.
    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            classes_num: int,
            loss_fn: Callable,
            optimizer_fn: Callable,
            measurement_op: Callable = qml.PauliZ
    ):
        if classes_num == 2:
            measurements_num = 1
        else:
            measurements_num = classes_num
        self.classes_num = classes_num

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )

    def predict_proba(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predict the probability of each class for the given input data. The output probabilities
        indicate the likelihood of each class for each sample.

        Args:
            x: An array of input data.

        Returns:
            An array of predicted probabilities. For binary classification, this will be a 1D array with
            a single probability for each sample. For multi-class classification, this will be a 2D array
            where each row corresponds to a sample and each column corresponds to a class.
        """
        logits = self.model(q_weights=self.params.q_weights, x_data=x)

        if self.classes_num == 2:
            return jnp.array(logits.ravel())
        else:
            return jnp.array(logits).T

    def predict(
            self,
            x: jnp.ndarray,
            threshold: float = 0.5
    ) -> jnp.ndarray:
        """
        Predict class labels for the given input data.

        For binary classification, the function applies a threshold to the output probabilities to
        determine the class labels. For multi-class classification, the function assigns each sample
         to the class with the highest probability.

        Args:
            x: An array of input data
            threshold: The threshold for converting probabilities to binary class labels. Defaults to 0.5.

        Returns:
            An array of predicted class labels. For binary classification, this will be a 1D array with
            binary labels (0 or 1). For multi-class classification, this will be a 1D array where each
            element is the predicted class index.
        """
        logits = self.predict_proba(x)

        if self.classes_num == 2:
            return jnp.where(logits >= threshold, 1, 0)
        else:
            return jnp.argmax(logits, axis=1)
