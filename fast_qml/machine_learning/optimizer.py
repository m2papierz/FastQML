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

import fast_qml
from fast_qml import QubitDevice


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
    """

    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int
    ):
        self._params = params
        self._q_node = q_node
        self._loss_fun = loss_fn
        self._epochs_num = epochs_num
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        self._np_module = self._get_numpy_module()

    @staticmethod
    def _get_numpy_module():
        """
        Determines and returns the appropriate numpy module based on the execution device.

        Returns:
            module: The numpy module corresponding to the current quantum device.
        """
        if fast_qml.DEVICE == QubitDevice.CPU.value:
            return importlib.import_module('pennylane.numpy')
        elif fast_qml.DEVICE == QubitDevice.CPU_JAX.value:
            return importlib.import_module('jax.numpy')
        else:
            NotImplementedError()

    @property
    def weights(self):
        """ Property to get the current model parameters. """
        return self._params

    def batch_generator(
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

    @abstractmethod
    def optimize(self, data, targets, verbose):
        """
        Abstract method for the optimization loop.

        This method should be implemented by subclasses to define the specific
        optimization algorithm.

        Args:
            data: Input data for the model.
            targets: Target data for training.
            verbose: Flag for verbose output during training.
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
            loss_fn: Callable
    ):
        super().__init__(
            params=params,
            q_node=q_node,
            loss_fn=loss_fn,
            epochs_num=epochs_num,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        self._opt = qml.AdamOptimizer(self._learning_rate)

    def optimize(self, data, targets, verbose):
        """
        Optimization loop.

        Args:
            data: Input data for the model.
            targets: Target data for training.
            verbose: Flag to control verbosity.
        """

        if data.shape[0] != targets.shape[0]:
            raise ValueError(
                "Data and targets must have the same number of samples"
            )

        for epoch in range(self._epochs_num):
            if self._batch_size:
                total_loss, num_batches = 0, 0
                for batch_data, batch_targets in self.batch_generator(data, targets):
                    self._params = self._opt.step(
                        self._calculate_loss, self._params, x_data=batch_data, y_data=batch_targets)
                    loss_val = self._calculate_loss(self._params, data, targets)
                    total_loss += loss_val
                    num_batches += 1
                train_loss = total_loss / num_batches
            else:
                self._params = self._opt.step(
                    self._calculate_loss, self._params, x_data=data, y_data=targets)
                train_loss = self._calculate_loss(self._params, data, targets)

            if verbose:
                print(f"Epoch {epoch + 1}/{self._epochs_num} - Training Loss: {train_loss:.5f}")

        return self._params


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
            batch_size: int = None
    ):
        super().__init__(
            params=params,
            q_node=q_node,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate
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

    def optimize(
            self,
            data: np.ndarray,
            targets: np.ndarray,
            verbose: bool
    ) -> None:
        """
        Optimization loop.

        Args:
            data: Input data for the model.
            targets: Target data for training.
            verbose: Flag to control verbosity.
        """
        if data.shape[0] != targets.shape[0]:
            raise ValueError(
                "Data and targets must have the same number of samples"
            )

        opt_state = self._opt.init(self._params)

        for epoch in range(self._epochs_num):
            if self._batch_size:
                total_loss, num_batches = 0, 0
                for batch_data, batch_targets in self.batch_generator(data, targets):
                    self._params, opt_state, loss_val = self._update_step(
                        self._params, opt_state, batch_data, batch_targets)
                    total_loss += loss_val
                    num_batches += 1
                train_loss = total_loss / num_batches
            else:
                self._params, opt_state, train_loss = self._update_step(
                    self._params, opt_state, data, targets)

            if verbose:
                print(f"Epoch {epoch + 1}/{self._epochs_num} - Training Loss: {train_loss:.5f}")
