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

import fast_qml
from fast_qml import QubitDevice
from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.machine_learning.optimizer import DefaultOptimizer, JITOptimizer
from fast_qml.machine_learning.loss_functions import MSELoss


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
        _interface: The computational interface (e.g., 'auto', 'jax') used by the quantum device.
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
        self._loss_fn = loss_fn
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num

        self._setup_device_and_optimizer()
        self._weights = self._initialize_weights()

    def _setup_device_and_optimizer(self):
        """
        Set up the quantum device and optimizer based on the current device configuration.
        """
        if fast_qml.DEVICE == QubitDevice.CPU.value:
            self._interface = 'auto'
            self._optimizer = DefaultOptimizer
            self._device = qml.device("default.qubit", wires=self._n_qubits)
        elif fast_qml.DEVICE == QubitDevice.CPU_JAX.value:
            self._interface = 'jax'
            self._optimizer = JITOptimizer
            self._optimizer.register_pytree_node()
            self._device = qml.device("default.qubit.jax", wires=self._n_qubits)
        else:
            raise NotImplementedError("The specified device is not supported.")

    @staticmethod
    def _is_valid_measurement_op(measurement_op):
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

    @abstractmethod
    def _initialize_weights(self) -> np.ndarray:
        """
        Initialize weights for the quantum circuit.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _quantum_layer(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> None:
        """
        Define and apply the quantum layer of the circuit.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _q_model(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ) -> qml.qnode:
        """
        Define and apply the quantum model for the variational estimator.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def fit(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            batch_size: int,
            verbose: bool
    ) -> np.ndarray:
        """
        Fit the model to the given data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
