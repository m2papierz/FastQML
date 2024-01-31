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

import warnings
from typing import Callable, Union, Tuple
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
        """
        Initialize weights for the quantum circuit.
        """
        weights = 0.1 * qml.numpy.random.random(
            self._ansatz.params_num, requires_grad=True)
        return weights

    def _quantum_layer(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> None:
        """
        Applies the quantum layer consisting of the feature map and the variational form.

        This method encodes classical data into quantum states using the specified feature map and
        then applies the variational form with the given weights.

        Args:
            weights: Parameters for the variational form.
            x_data: Classical data to be encoded into quantum states.
        """
        self._feature_map.apply(features=x_data)
        self._ansatz.apply(params=weights)

    def _q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> qml.qnode:
        """
        Defines the quantum model circuit to be used in optimization.

        This method creates a PennyLane QNode that represents the quantum circuit. It applies the quantum layer
        to the input data and then performs the specified measurements.

        Args:
            weights: Parameters for the variational form.
            x_data: Classical data to be encoded into quantum states.

        Returns:
            A PennyLane QNode that outputs the expectation values of the measurement operators.
        """
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
            batch_size: int = None,
            verbose: bool = True
    ) -> None:
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

        self._weights = optimizer.weights


class VQRegressor(VariationalQuantumEstimator):
    """
    A variational quantum estimator specifically designed for regression tasks. It inherits from
    VariationalQuantumEstimator and ensures that the loss function used is appropriate for regression.

    This class is suitable for tasks where the goal is to predict continuous or quantitative outputs.
    """

    _allowed_losses = [MSELoss, HuberLoss, LogCoshLoss]

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable = MSELoss(),
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

    def predict(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Returns the predictions (model outputs) for the given input data.

        Args:
            x: An array of input data.
        """
        return np.array(
            [self._q_model(weights=self._weights, x_data=_x) for _x in x]
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

    _allowed_losses = [MSELoss, HuberLoss, LogCoshLoss, BinaryCrossEntropyLoss]

    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            classes_num: int,
            loss_fn: Callable = None,
            measurement_op: Callable = qml.PauliZ
    ):
        self._validate_loss_fn(loss_fn)
        self._classes_num = classes_num

        loss_fn, measurements_num = self._set_loss_function(
            classes_num=classes_num, loss_fn=loss_fn)

        super().__init__(
            n_qubits=n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            loss_fn=loss_fn,
            measurement_op=measurement_op,
            measurements_num=measurements_num
        )

    def _validate_loss_fn(
            self,
            loss_fn: Callable
    ) -> None:
        """
        Validates the provided loss function.
        """
        if loss_fn is not None and not any(isinstance(loss_fn, loss) for loss in self._allowed_losses):
            raise AttributeError("Invalid loss function.")

    @staticmethod
    def _set_loss_function(
            classes_num,
            loss_fn
    ) -> Union[Tuple[Callable, int]]:
        """
        Selects the appropriate loss function based on the number of classes.
        """
        if classes_num == 2:
            return BinaryCrossEntropyLoss(), 1
        elif classes_num > 2:
            if loss_fn is not None:
                warnings.warn(
                    "For multi-class classification (classes_num > 2), the provided "
                    "loss_fn will be ignored, and CrossEntropyLoss will be used instead.",
                    category=UserWarning
                )
            return CrossEntropyLoss(), classes_num
        else:
            raise ValueError("Classes must be 2 or more.")

    def predict_proba(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Predict the probability of each class for the given input data.nThe output probabilities
        indicate the likelihood of each class for each sample.

        Args:
            x: An array of input data.

        Returns:
            An array of predicted probabilities. For binary classification, this will be a 1D array with
            a single probability for each sample. For multi-class classification, this will be a 2D array
            where each row corresponds to a sample and each column corresponds to a class.
        """
        probabilities = [
            self._q_model(self._weights, np.array([sample]))
            for sample in x
        ]
        return np.array(probabilities).ravel()

    def predict(
            self,
            x: np.ndarray,
            threshold: float = 0.5
    ) -> np.ndarray:
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
        predictions = self.predict_proba(x)

        if self._classes_num == 2:
            return np.where(predictions >= threshold, 1, 0)
        else:
            return np.argmax(predictions, axis=1)
