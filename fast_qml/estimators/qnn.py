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
This module introduces classes for constructing quantum neural networks (QNNs). It defines a base class
`QNN` for general quantum neural networks, along with specialized subclasses `QNNRegressor` for regression
tasks and `QNNClassifier` for classification tasks.

The QNNs are designed with flexibility to accommodate various quantum feature maps, variational forms, and
loss functions, making them suitable for a wide range of quantum machine learning applications.

Classes:
- QNN: Base class for creating quantum neural networks.
- QNNRegressor: Subclass for regression tasks using quantum neural networks.
- QNNClassifier: Subclass for classification tasks using quantum neural networks.
"""

from typing import (
    Union, Dict, Any, Tuple, Callable)

import jax
import numpy as np
import pennylane as qml
from jax import numpy as jnp

from fast_qml.quantum_circuits.feature_maps import FeatureMap, AmplitudeEmbedding
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.core.estimator import QuantumEstimator


class QNN(QuantumEstimator):
    """
    Base class for quantum neural networks, providing the essential components to construct
    a quantum circuit for machine learning tasks.

    This class supports data reuploading, allowing for a flexible encoding of input data into
    the quantum circuit, and multiple layers of variational forms for complex model architectures.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        feature_map: Quantum feature map for encoding input data.
        ansatz: Variational form for the quantum circuit.
        layers_num: Number of layers in the quantum circuit.
        measurement_op: Measurement operator for extracting information from the circuit.
        loss_fn: Loss function for model training.
        measurements_num: Number of wires on which to run measurements.
        data_reuploading: Flag to enable or disable data reuploading.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer: Callable,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            measurements_num: int = 1,
            data_reuploading: bool = False
    ):
        self._layers_num = layers_num
        self._data_reuploading = data_reuploading

        if self._data_reuploading and isinstance(feature_map, AmplitudeEmbedding):
            raise ValueError(
                "Data reuploading is not compatible with Amplitude Embedding ansatz. PennyLane "
                "does not allow to use multiple state preparation operations at the moment."
            )

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            optimizer=optimizer,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )

    def _initialize_parameters(
            self
    ) -> Union[jnp.ndarray, Dict[str, Any]]:
        """
        Initialize weights for the quantum circuit.
        """
        key = jax.random.PRNGKey(42)
        params_shape = self._ansatz.params_num
        if isinstance(params_shape, int):
            weights = 0.1 * jax.random.normal(
                key, shape=(self._layers_num, self._ansatz.params_num))
        else:
            weights = 0.1 * jax.random.normal(
                key, shape=(self._layers_num, *self._ansatz.params_num))
        return weights

    def _quantum_layer(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray
    ) -> None:
        """
        Applies the quantum layer, which includes the feature map and the ansatz, to the circuit.

        This method is conditional on the data reuploading flag. If data reuploading is enabled,
        the feature map is applied at every layer.

        Args:
            weights: Parameters for the variational form.
            x_data: Input data to be encoded into the quantum state.
        """
        if not self._data_reuploading:
            self._ansatz.apply(params=weights)
        else:
            self._feature_map.apply(features=x_data)
            self._ansatz.apply(params=weights)

    def q_model(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray
    ) -> qml.qnode:
        """
        Defines the quantum model circuit for the neural network.

        This method creates a PennyLane QNode that constructs the quantum circuit by applying
        the quantum layer multiple times based on the number of layers specified.

        Args:
            weights: Parameters for the variational form.
            x_data: Input data to be processed by the quantum circuit.

        Returns:
            A PennyLane QNode representing the quantum circuit.
        """
        @jax.jit
        @qml.qnode(device=self._device, interface="jax")
        def _circuit():
            if not self._data_reuploading:
                self._feature_map.apply(features=x_data)

            for i in range(self._layers_num):
                self._quantum_layer(
                    weights=weights[i], x_data=x_data)

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

        def draw_circuit(params, inputs):
            if not self._data_reuploading:
                self._feature_map.apply(features=inputs)

            for i in range(self._layers_num):
                self._quantum_layer(
                    weights=params[i], x_data=inputs)

        aux_input = np.array([aux_input])
        print(qml.draw(draw_circuit)(self.params, aux_input))


class QNNRegressor(QNN):
    """
    A quantum neural network specialized for regression tasks. It inherits from the QNN class
    and ensures the use of loss functions suitable for regression.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer: Callable,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            data_reuploading: bool = False
    ):
        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            layers_num=layers_num,
            measurement_op=measurement_op,
            loss_fn=loss_fn,
            optimizer=optimizer,
            measurements_num=1,
            data_reuploading=data_reuploading
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
            self.q_model(weights=self.params, x_data=x)
        ).ravel()


class QNNClassifier(QNN):
    """
    A quantum neural network designed for classification tasks. Depending on the number of
    classes, it automatically selects a suitable loss function (BinaryCrossEntropyLoss for binary
    classification or CrossEntropyLoss for multi-class classification).

    This class is ideal for tasks where the goal is to categorize inputs into discrete classes.

    Args:
        classes_num: Number of classes for the classification task.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer: Callable,
            classes_num: int,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            data_reuploading: bool = False
    ):
        if classes_num == 2:
            measurements_num = 1
        else:
            measurements_num = classes_num
        self.num_classes = classes_num

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            layers_num=layers_num,
            measurement_op=measurement_op,
            loss_fn=loss_fn,
            optimizer=optimizer,
            measurements_num=measurements_num,
            data_reuploading=data_reuploading
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
        logits = jnp.array(self.q_model(self.params, x))
        if self.num_classes == 2:
            return logits.ravel()
        else:
            return logits.T

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

        if self.num_classes == 2:
            return jnp.where(logits >= threshold, 1, 0)
        else:
            return jnp.argmax(logits, axis=1)
