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
from typing import (
    Callable, Union, Any, Tuple, Dict, Mapping)

import jax
import torch
import flax.linen as nn
import numpy as np
import pennylane as qml
from jax import numpy as jnp
from torch.utils.data import DataLoader

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.core.callbacks import EarlyStopping
from fast_qml.machine_learning.optimizer import HybridOptimizer

from fast_qml.core.optimizer import QuantumOptimizer
from fast_qml.core.opt_classical import ClassicalOptimizer


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
        loss_fn: The loss function used for core.
        measurements_num: Number of wires on which to run measurements.
    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            measurement_op: Callable = qml.PauliZ,
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
            weights: jnp.ndarray,
            x_data: jnp.ndarray
    ) -> qml.qnode:
        """
        Define and apply the quantum model for the variational estimator.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def fit(
            self,
            train_data: Union[np.ndarray, torch.Tensor, DataLoader],
            val_data: Union[np.ndarray, torch.Tensor, DataLoader],
            train_targets: Union[np.ndarray, torch.Tensor, None] = None,
            val_targets: Union[np.ndarray, torch.Tensor, None] = None,
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
            train_data: Input features for training.
            train_targets: Target outputs for training.
            val_data: Input features for validation.
            val_targets: Target outputs for validation.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            early_stopping: Instance of EarlyStopping to be used during training.
            verbose : If True, prints verbose messages during training.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        optimizer = QuantumOptimizer(
            c_params=None,
            q_params=self.weights,
            batch_stats=None,
            model=self.q_model,
            loss_fn=self.loss_fn,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping=early_stopping
        )

        optimizer.optimize(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            epochs_num=num_epochs,
            verbose=verbose
        )

        self.weights = optimizer.weights


class ClassicalEstimator:
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            batch_norm: bool
    ):
        self._c_model = c_model
        self._loss_fn = loss_fn
        self._batch_norm = batch_norm

        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)
        self._params = self._initialize_parameters(
            input_shape=input_shape, batch_norm=batch_norm)

    @abstractmethod
    def _initialize_parameters(
            self,
            input_shape: Union[int, Tuple[int]],
            batch_norm: bool = False
    ) -> Dict[str, Any]:
        """
        Initializes parameters for the classical and quantum models.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _model(
            self,
            weights: Dict[str, Mapping[str, jnp.ndarray]],
            x_data: jnp.ndarray,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            training: bool
    ):
        """
        Defines the classical model inference.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def fit(
            self,
            train_data: np.ndarray,
            train_targets: np.ndarray,
            val_data: np.ndarray,
            val_targets: np.ndarray,
            learning_rate: float = 0.01,
            num_epochs: int = 500,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            verbose: bool = True
    ) -> None:
        """
        Trains the hybrid classical-quantum model on the given data.

        This method optimizes the weights of the hybrid model using the specified loss function
        and optimizer. It updates the weights based on the training data over a number of epochs.

        Args:
            train_data: Input features for training.
            train_targets: Target outputs for training.
            val_data: Input features for validation.
            val_targets: Target outputs for validation.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            early_stopping: Instance of EarlyStopping to be used during training.
            verbose : If True, prints verbose messages during training.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        if self._batch_norm:
            weights, batch_stats = (
                self._params['weights'], self._params['batch_stats'])
        else:
            weights, batch_stats = self._params, None

        optimizer = ClassicalOptimizer(
            c_params=weights,
            q_params=None,
            batch_stats=batch_stats,
            model=self._model,
            loss_fn=self._loss_fn,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping=early_stopping
        )

        optimizer.optimize(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            epochs_num=num_epochs,
            verbose=verbose
        )

        self._params['weights'] = optimizer.weights


class HybridEstimator:
    """
    Base class for creating hybrid quantum-classical machine learning models.

    This class combines classical neural network models with quantum neural networks or variational
    quantum algorithms to form a hybrid model for regression or classification tasks.

    Attributes:
        _c_model: The classical neural network model.
        _q_model: The quantum model, which can be either a quantum neural network or a variational quantum algorithm.
        _optimizer: Optimizer for training the hybrid model.
        _inp_rng: Random number generator key for input initialization.
        _init_rng: Random number generator key for model initialization.
        _weights: Initialized weights for both classical and quantum models.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum model.
    """
    def __init__(
            self,
            input_shape,
            c_model,
            q_model,
    ):
        self._c_model = c_model
        self._q_model = q_model

        self._optimizer = HybridOptimizer

        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)
        self._weights = self._initialize_weights(input_shape)

    def _initialize_weights(
            self,
            input_shape: Union[int, Tuple[int]]
    ) -> Dict[str, Any]:
        """
        Initializes weights for the classical and quantum models.

        Args:
            input_shape: The shape of the input data.

        Returns:
            A dictionary containing initialized weights for both classical and quantum models.
        """
        c_inp = jax.random.normal(self._inp_rng, shape=input_shape)
        c_weights = self._c_model.init(self._init_rng, c_inp)
        return {
            'c_weights': c_weights,
            'q_weights': self._q_model.weights
        }

    def _model(
            self,
            c_weights: Dict[str, Mapping[str, jnp.ndarray]],
            q_weights: jnp.ndarray,
            x_data: jnp.ndarray
    ):
        """
        Defines the hybrid model by combining classical and quantum models.

        Args:
            c_weights: Weights of the classical model.
            q_weights: Weights of the quantum model.
            x_data: Input data for the model.

        Returns:
            The output of the hybrid model.
        """
        def _hybrid_model():
            c_out = self._c_model.apply(c_weights, x_data)
            c_out = jax.numpy.array(c_out)
            q_out = self._q_model.q_model(weights=q_weights, x_data=c_out)
            return q_out

        return _hybrid_model()

    def fit(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray = None,
            y_val: jnp.ndarray = None,
            learning_rate: float = 0.01,
            num_epochs: int = 500,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            verbose: bool = True
    ) -> None:
        """
        Trains the hybrid classical-quantum model on the given data.

        This method optimizes the weights of the hybrid model using the specified loss function
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
            c_params=self._weights['c_weights'],
            q_params=self._weights['q_weights'],
            model=self._model,
            loss_fn=self._q_model.loss_fn,
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

        self._weights['c_weights'], self._weights['q_weights'] = optimizer.weights
