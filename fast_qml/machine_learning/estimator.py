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
from abc import abstractmethod

import numpy as np
import pennylane as qml
from jax import numpy as jnp

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.optimizer import QuantumOptimizer
from fast_qml.machine_learning.loss_functions import MSELoss
from fast_qml.machine_learning.callbacks import EarlyStopping


class QuantumEstimator:
    """
    Base class for creating quantum estimators.

    This class provides a framework for quantum machine learning models,
    and is intended to be subclassed for specific implementations.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        feature_map: The feature map for encoding classical data into quantum states.
        ansatz: The variational form (ansatz) for the quantum circuit.
        measurement_op: The measurement operator or observable used in the circuit.
        loss_fn: The loss function used for optimization.
        measurements_num: Number of wires on which to run measurements.

    Attributes:
        _optimizer: The optimizer used for training the quantum circuit.
        _device: The quantum device on which the circuit will be executed.
    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_op: Callable = qml.PauliZ,
            loss_fn: Callable = MSELoss(),
            measurements_num: int = 1
    ):
        # Validate measurement operation
        if not self._is_valid_measurement_op(measurement_op):
            raise ValueError("Invalid measurement operation provided.")

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self.loss_fn = loss_fn
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num

        self._device = qml.device(
            name="default.qubit.jax", wires=self._n_qubits)
        self._optimizer = QuantumOptimizer

        self.weights = self._initialize_weights()

    @staticmethod
    def _is_valid_measurement_op(measurement_op):
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

    @abstractmethod
    def _initialize_weights(self) -> jnp.ndarray:
        """
        Initialize weights for the quantum circuit.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> qml.qnode:
        """
        Define and apply the quantum model for the variational estimator.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
            learning_rate: float = 0.01,
            num_epochs: int = 500,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            verbose: bool = True
    ) -> None:
        """
        Trains the variational quantum estimator on the provided dataset.

        This method optimizes the weights of the variational circuit using the specified loss function
        and optimizer. It updates the weights based on the training data over a number of epochs.

        Args:
            x_train: Input features for training.
            y_train: Target outputs for training.
            x_val: Input features for validation.
            y_val: Target outputs for validation.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            early_stopping: Instance of EarlyStopping to be used during training.
            verbose : If True, prints verbose messages during training.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        optimizer = self._optimizer(
            c_params=None,
            q_params=self.weights,
            model=self.q_model,
            loss_fn=self.loss_fn,
            batch_size=batch_size,
            epochs_num=num_epochs,
            learning_rate=learning_rate,
            early_stopping=early_stopping
        )

        optimizer.optimize(
            x_train=jnp.array(x_train),
            y_train=jnp.array(y_train),
            x_val=jnp.array(x_val),
            y_val=jnp.array(y_val),
            verbose=verbose
        )

        self.weights = optimizer.weights
