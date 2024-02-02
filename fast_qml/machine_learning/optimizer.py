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
Quantum Machine Learning Optimization Module

This module provides classes for implementing various optimization algorithms specifically designed for
quantum machine learning models. These optimizers are tailored to work with quantum nodes and are capable
of handling both batch and non-batch training scenarios.

Classes:
    - Optimizer: Base class for quantum model optimizers.
    - DefaultOptimizer: Standard optimizer using the Adam algorithm from Pennylane.
    - JITOptimizer: Optimizer utilizing JAX for JIT compilation, enhancing performance on CPU.
"""

import importlib
from typing import Callable, Tuple
from abc import abstractmethod

import jax
import optax
import numpy as np
import pennylane as qml

from jax import tree_util
from jax.example_libraries.optimizers import OptimizerState

from fast_qml import device_manager, QubitDevice
from fast_qml.machine_learning.callbacks import EarlyStopping, BestModelCheckpoint


class Optimizer:
    """
    Base class for optimizers used in quantum machine learning models.

    This class provides the foundational structure for implementing various optimization algorithms.
    It is designed to integrate with quantum nodes (q_nodes) and supports dynamic numpy module
    selection based on the quantum device in use.

    Attributes:
        _params: Parameters of the quantum model.
        _q_node: Quantum node representing the quantum circuit.
        _loss_fun: Loss function used for optimization.
        _epochs_num: Number of training epochs.
        _batch_size: Batch size for training.
        _learning_rate: Learning rate.
        _np_module: Numpy module (Pennylane or JAX), depending on the execution device.
        _early_stopping: Instance of EarlyStopping to be used during training.
    """

    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        self._params = params
        self._q_node = q_node
        self._loss_fun = loss_fn
        self._epochs_num = epochs_num
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        self._early_stopping = early_stopping
        self._np_module = self._get_numpy_module()

        self._best_model_checkpoint = None
        if retrieve_best_weights:
            self._best_model_checkpoint = BestModelCheckpoint()

    @staticmethod
    def _get_numpy_module():
        """
        Determines and returns the appropriate numpy module based on the execution device.

        Returns:
            module: The numpy module corresponding to the current quantum device.
        """
        if device_manager.device == QubitDevice.CPU.value:
            return importlib.import_module('pennylane.numpy')
        elif device_manager.device == QubitDevice.CPU_JAX.value:
            return importlib.import_module('jax.numpy')
        else:
            NotImplementedError()

    @property
    def weights(self):
        """ Property to get the current model parameters. """
        return self._params

    def _batch_generator(
            self,
            data: np.ndarray,
            targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator to yield batches of data and targets for training.

        Args:
            data: Input data for the model.
            targets: Target data for training.

        Yields:
            Tuple: A tuple of a data batch and a target batch.
        """
        for i in range(0, len(data), self._batch_size):
            yield data[i:i + self._batch_size], targets[i:i + self._batch_size]

    def _calculate_loss(
            self,
            weights: np.ndarray,
            x_data: np.ndarray,
            y_data: np.ndarray
    ):
        """
        Calculates the loss for a given set of weights, input data, and target data.

        Args:
            weights: Parameters of the quantum model.
            x_data: Input data for the model.
            y_data: Target data for training.

        Returns:
            Computed loss value.
        """
        predictions = self._np_module.array(
            self._q_node(weights=weights, x_data=x_data))
        return self._loss_fun(y_real=y_data, y_pred=predictions)

    @staticmethod
    def _validate_data(
            x_data: np.ndarray,
            y_data: np.ndarray,
            data_type: str
    ) -> None:
        """
        Validates that the input and target data have the same number of samples.

        Args:
            x_data: Input data.
            y_data: Target data.
            data_type: A string indicating the type of data (Training or Validation).
        """
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"{data_type} data and targets must have the same number of samples"
            )

    @abstractmethod
    def optimize(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            verbose: bool
    ) -> None:
        """
        Abstract method for the optimization loop.

        This method should be implemented by subclasses to define the specific
        optimization algorithm.

        Args:
            x_train: Training input data for the model.
            y_train: Training target data.
            x_val: Validation input data.
            y_val: Validation target data.
            verbose: Flag to control verbosity.
        """
        pass


class DefaultOptimizer(Optimizer):
    """
    Default optimizer extending the base Optimizer class.

    This optimizer implements a standard optimization procedure using the Adam optimizer
    from Pennylane. It is designed for quantum machine learning models and provides functionality
    for both batch and non-batch training.

    Class provides functionality for both batch and non-batch training - if batch_size is given
    then batch training is started, otherwise non-batching training runs.

    Attributes:
        _opt: Instance of the Adam optimizer from the qml library.
    """

    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            learning_rate: float,
            epochs_num: int,
            batch_size: int,
            loss_fn: Callable,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            params=params,
            q_node=q_node,
            loss_fn=loss_fn,
            epochs_num=epochs_num,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )
        self._opt = qml.AdamOptimizer(self._learning_rate)

    def _perform_training_epoch(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray
    ) -> float:
        """
        Performs a single training epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.

        Returns:
            Average training loss for the epoch.
        """

        if self._batch_size:
            total_loss, num_batches = 0.0, 0
            for batch_data, batch_targets in self._batch_generator(x_train, y_train):
                self._params = self._opt.step(
                    self._calculate_loss, self._params, x_data=batch_data, y_data=batch_targets)
                batch_loss = self._calculate_loss(self._params, batch_data, batch_targets)
                total_loss += batch_loss
                num_batches += 1
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            self._params = self._opt.step(
                self._calculate_loss, self._params, x_data=x_train, y_data=y_train)
            average_loss = self._calculate_loss(self._params, x_train, y_train)

        return average_loss

    def _perform_validation_epoch(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Performs a validation step.

        Args:
            x_val: Validation input data.
            y_val: Validation target data.

        Returns:
            Average validation loss.
        """
        if self._batch_size:
            total_loss, num_batches = 0.0, 0
            for batch_data, batch_targets in self._batch_generator(x_val, y_val):
                batch_loss = self._calculate_loss(self._params, batch_data, batch_targets)
                total_loss += batch_loss
                num_batches += 1
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            average_loss = self._calculate_loss(self._params, x_val, y_val)
        return average_loss

    def optimize(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
            verbose: bool = True
    ) -> None:
        """
        Optimization loop with validation.

        Args:
            x_train: Training input data for the model.
            y_train: Training target data.
            x_val: Optional validation input data.
            y_val: Optional validation target data.
            verbose: Flag to control verbosity.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        self._validate_data(x_train, y_train, data_type='Training')
        if x_val is not None and y_val is not None:
            self._validate_data(x_val, y_val, data_type='Validation')

        message = ""
        for epoch in range(self._epochs_num):
            training_loss = self._perform_training_epoch(x_train=x_train, y_train=y_train)

            if verbose:
                message = f"Epoch {epoch + 1}/{self._epochs_num} - train_loss: {training_loss:.5f}"

            if x_val is not None and y_val is not None:
                validation_loss = self._perform_validation_epoch(x_val=x_val, y_val=y_val)

                if self._best_model_checkpoint:
                    self._best_model_checkpoint.update(self._params, validation_loss)

                if verbose:
                    message += f", val_loss: {validation_loss:.5f}"

                # Early stopping logic
                if self._early_stopping:
                    self._early_stopping(validation_loss)
                    if self._early_stopping.stop_training:
                        if verbose:
                            print(f"Stopping early at epoch {epoch + 1}.")
                        break

            # Load best model parameters at the end of training
            if self._best_model_checkpoint:
                self._best_model_checkpoint.load_best_model(self)

            if verbose:
                print(message)


class JITOptimizer(Optimizer):
    """
    JIT Optimizer that extends from a base Optimizer class.

    This optimizer is specifically designed to use JAX for Just-In-Time compilation to speed
    up the optimization process on CPU. It integrates with quantum nodes (q_nodes) and employs
    an Adam optimizer.

    Class provides functionality for both batch and non-batch training - if batch_size is given
    then batch training is started, otherwise non-batching training runs.

    Args:
        params: Initial parameters of the model.
        q_node: Quantum node representing the quantum circuit.
        loss_fn: Loss function for the optimization.
        epochs_num: Number of epochs for the training.
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Size of batches for training. If None, batch processing is not used.
    """
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            params=params,
            q_node=q_node,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )
        self._opt = optax.adam(learning_rate=learning_rate)

    def _tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.

        This method is used for JAX's automatic differentiation and
        is required for the class to work with jax.jit optimizations.

        Returns:
            Tuple: Contains the parameters as children and other attributes as auxiliary data.
        """
        children = (self._params,)
        aux_data = {
            'q_node': self._q_node,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size,
            'epochs_num': self._epochs_num,
            'loss_fn': self._loss_fun
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
       Reconstructs the class instance from JAX tree operations.

       Args:
           aux_data: Auxiliary data containing the attributes of the instance.
           children: Contains the parameters of the model.

       Returns:
           JITOptimizer: A new instance of JITOptimizer reconstructed from the tree data.
       """
        return cls(
            *children,
            q_node=aux_data['q_node'],
            learning_rate=aux_data['learning_rate'],
            batch_size=aux_data['batch_size'],
            epochs_num=aux_data['epochs_num'],
            loss_fn=aux_data['loss_fn']
        )

    @classmethod
    def register_pytree_node(cls):
        """
        Registers the JITOptimizer class as a JAX pytree node.

        This method allows JAX to recognize JITOptimizer instances as pytrees,
        enabling automatic differentiation and other JAX functionalities.
        """
        tree_util.register_pytree_node(
            nodetype=cls,
            flatten_func=cls._tree_flatten,
            unflatten_func=cls._tree_unflatten
        )

    @jax.jit
    def _update_step(
            self,
            params: np.ndarray,
            opt_state: OptimizerState,
            data: np.ndarray,
            targets: np.ndarray
    ) -> Tuple[np.ndarray, OptimizerState, np.ndarray]:
        """
        Perform a single update step.

        Args:
            params: Model parameters.
            opt_state: Pytree representing the optimizer state to be updated.
            data: Input data.
            targets: Target data.

        Returns:
            Updated parameters, optimizer state, and loss value.
        """
        loss_val, grads = jax.value_and_grad(
            self._calculate_loss)(params, data, targets)
        updates, opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    @jax.jit
    def _validation_step(
            self,
            params: np.ndarray,
            data: np.ndarray,
            targets: np.ndarray
    ) -> np.ndarray:
        """
        Perform a validation step.

        Args:
            params: Model parameters.
            data: Input data.
            targets: Target data.

        Returns:
            Loss value for the given data and targets.
        """
        return self._calculate_loss(params, data, targets)

    def _perform_training_epoch(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            optimizer_state: OptimizerState
    ) -> Tuple[float, OptimizerState]:
        """
        Performs a single training epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            optimizer_state: Current state of the optimizer.

        Returns:
            Average training loss for the epoch.
        """

        if self._batch_size:
            total_loss, num_batches = 0.0, 0
            for batch_data, batch_targets in self._batch_generator(x_train, y_train):
                self._params, optimizer_state, batch_loss = self._update_step(
                    self._params, optimizer_state, batch_data, batch_targets)
                total_loss += batch_loss
                num_batches += 1
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            self._params, optimizer_state, average_loss = self._update_step(
                self._params, optimizer_state, x_train, y_train)

        return average_loss, optimizer_state

    def _perform_validation_epoch(
            self,
            x_val: np.ndarray,
            y_val: np.ndarray
    ) -> float:
        """
        Performs a validation step.

        Args:
            x_val: Validation input data.
            y_val: Validation target data.

        Returns:
            Average validation loss.
        """
        if self._batch_size:
            total_loss, num_batches = 0.0, 0
            for batch_data, batch_targets in self._batch_generator(x_val, y_val):
                batch_loss = self._validation_step(self._parameters, batch_data, batch_targets)
                total_loss += batch_loss
                num_batches += 1
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            average_loss = self._validation_step(self._params, x_val, y_val)
        return average_loss

    def optimize(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
            verbose: bool = True
    ) -> None:
        """
        Optimization loop with validation.

        Args:
            x_train: Training input data for the model.
            y_train: Training target data.
            x_val: Optional validation input data.
            y_val: Optional validation target data.
            verbose: Flag to control verbosity.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        self._validate_data(x_train, y_train, data_type='Training')
        if x_val is not None and y_val is not None:
            self._validate_data(x_val, y_val, data_type='Validation')

        opt_state = self._opt.init(self._params)

        message = ""
        for epoch in range(self._epochs_num):
            training_loss, opt_state = self._perform_training_epoch(
                x_train=x_train, y_train=y_train, optimizer_state=opt_state)

            if verbose:
                message = f"Epoch {epoch + 1}/{self._epochs_num} - train_loss: {training_loss:.5f}"

            if x_val is not None and y_val is not None:
                validation_loss = self._perform_validation_epoch(x_val=x_val, y_val=y_val)

                if self._best_model_checkpoint:
                    self._best_model_checkpoint.update(self._params, validation_loss)

                if verbose:
                    message += f", val_loss: {validation_loss:.5f}"

                # Early stopping logic
                if self._early_stopping:
                    self._early_stopping(validation_loss)
                    if self._early_stopping.stop_training:
                        if verbose:
                            print(f"Stopping early at epoch {epoch + 1}.")
                        break

            # Load best model parameters at the end of training
            if self._best_model_checkpoint:
                self._best_model_checkpoint.load_best_model(self)

            if verbose:
                print(message)
