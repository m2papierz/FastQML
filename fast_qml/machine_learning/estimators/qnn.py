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

from typing import Callable

import numpy as np
import pennylane as qml

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.estimator import QuantumEstimator
from fast_qml.machine_learning.loss_functions import (
    MSELoss, HuberLoss, LogCoshLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
)


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
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            loss_fn: Callable = MSELoss(),
            measurements_num: int = 1,
            data_reuploading: bool = False
    ):
        self._layers_num = layers_num

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )

        self._data_reuploading = data_reuploading

    def _initialize_weights(self) -> np.ndarray:
        """
        Initialize weights for the quantum circuit.
        """
        shape = (self._layers_num, self._ansatz.params_num)
        weights = 0.1 * qml.numpy.random.random(
            shape, requires_grad=True)
        return weights

    def _quantum_layer(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> None:
        """
        Applies the quantum layer, which includes the feature map and the ansatz, to the circuit.

        This method is conditional on the data reuploading flag. If data reuploading is enabled,
        the feature map is applied at every layer.

        Args:
            weights: Parameters for the variational form.
            x_data: Input data to be encoded into the quantum state.
        """
        if self._data_reuploading:
            self._feature_map.apply(features=x_data)
            self._ansatz.apply(params=weights)
        else:
            self._ansatz.apply(params=weights)

    def _q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> qml.qnode:
        """
        Defines the quantum model circuit for the neural network.

        This method creates a PennyLane QNode that constructs the quantum circuit by applying
        the quantum layer multiple times based on the number of layers specified.

        Args:
            weights: Parameters for the variational form.
            x_data: Input data to be processed by the quantum circuit.

        Returns:
            =A PennyLane QNode representing the quantum circuit.
        """
        @qml.qnode(device=self._device, interface=self._interface)
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

    def fit(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            batch_size: int = None,
            verbose: bool = True
    ) -> np.ndarray:
        """
        Trains the variational quantum estimator on the provided dataset.

        This method optimizes the weights of the variational circuit using the specified loss function
        and optimizer. It updates the weights based on the training data over a number of epochs.

        Args:
            x_data: Input features for training.
            y_data: Target outputs for training.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            verbose : If True, prints verbose messages during training.

        Returns:
            The optimized weights after training.
        """
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


class QNNRegressor(QNN):
    """
    A quantum neural network specialized for regression tasks. It inherits from the QNN class
    and ensures the use of loss functions suitable for regression.
    """

    _allowed_losses = [MSELoss, HuberLoss, LogCoshLoss]

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            loss_fn: Callable = MSELoss(),
            data_reuploading: bool = False
    ):
        if not any(isinstance(loss_fn, loss) for loss in self._allowed_losses):
            raise AttributeError("Invalid loss function.")

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            layers_num=layers_num,
            measurement_op=measurement_op,
            loss_fn=loss_fn,
            measurements_num=1,
            data_reuploading=data_reuploading
        )


class QNNClassifier(QNN):
    """
    A quantum neural network designed for classification tasks. Depending on the number of
    classes, it automatically selects a suitable loss function (BinaryCrossEntropyLoss for binary
    classification or CrossEntropyLoss for multi-class classification).

    This class is ideal for tasks where the goal is to categorize inputs into discrete classes.

    Args:
        classes_num (int): Number of classes for the classification task.
    """

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            classes_num: int,
            layers_num: int = 1,
            measurement_op: Callable = qml.PauliZ,
            data_reuploading: bool = False
    ):
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
            layers_num=layers_num,
            measurement_op=measurement_op,
            loss_fn=loss_fn,
            measurements_num=measurements_num,
            data_reuploading=data_reuploading
        )
