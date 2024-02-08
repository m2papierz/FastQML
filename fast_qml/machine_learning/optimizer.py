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
    - QuantumOptimizer: Optimizer designed to work with quantum-only models.
    - HybridOptimizer: Optimizer designed to work with quantum-classical models.
"""

from abc import abstractmethod
from typing import (
    Callable, Tuple, Union, Dict, Mapping
)

import jax
import optax
import numpy as np
import flax.linen as nn
import pennylane as qml

from jax.example_libraries.optimizers import OptimizerState
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from fast_qml.machine_learning.callbacks import EarlyStopping, BestModelCheckpoint


class Optimizer:
    """
    Base class for optimizers used in quantum machine learning models.

    This class provides the foundational structure for implementing various optimization
    algorithms. It is designed to integrate with both classical Flax deep learning models
    and quantum models.

    Attributes:
        _c_params: Parameters of the classical model.
        _q_params: Parameters of the quantum model.
        _model: Quantum node representing the quantum circuit.
        _loss_fun: Loss function used for optimization.
        _epochs_num: Number of training epochs.
        _batch_size: Batch size for training.
        _learning_rate: Learning rate.
        _early_stopping: Instance of EarlyStopping to be used during training.
        _best_model_checkpoint: Instance of BestModelCheckpoint to be used during training.
    """

    def __init__(
            self,
            c_params: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            q_params: Union[jnp.ndarray, None],
            model: [qml.qnode, Callable],
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        self._c_params = c_params
        self._q_params = q_params
        self._model = model
        self._loss_fun = loss_fn
        self._epochs_num = epochs_num
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        self._early_stopping = early_stopping

        self._best_model_checkpoint = None
        if retrieve_best_weights:
            self._best_model_checkpoint = BestModelCheckpoint()

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations. This method is used
        for JAX automatic differentiation and is required for the class to work
        with jax.jit optimizations.
        """
        children = [self._c_params, self._q_params]
        aux_data = {
            'model': self._model,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size,
            'epochs_num': self._epochs_num,
            'loss_fn': self._loss_fun
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
       Reconstructs the class instance from JAX tree operations.
       """
        return cls(*children, **aux_data)

    @property
    def weights(
            self
    ) -> Union[jnp.ndarray, Dict[str, Mapping[str, jnp.ndarray]], Tuple]:
        """
        Property to get the current model parameters.
        """
        # If both parameters are set return a tuple of both.
        if self._c_params is not None and self._q_params is not None:
            return self._c_params, self._q_params

        # If only classical parameters are set, return them.
        if self._c_params is not None:
            return self._c_params

        # If only quantum parameters are set, return them.
        if self._q_params is not None:
            return self._q_params

        # If neither parameter is set raise an error.
        raise ValueError("No model parameters are set.")

    def _batch_generator(
            self,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    @staticmethod
    def _validate_data(
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
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
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray,
            y_val: jnp.ndarray,
            verbose: bool
    ):
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
        return NotImplementedError("Subclasses must implement this method.")


class SingleModelOptimizer(Optimizer):
    def __init__(
            self,
            c_params: Union[jnp.ndarray, None],
            q_params: Union[jnp.ndarray, None],
            model: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            c_params=c_params,
            q_params=q_params,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )
        self._opt = optax.adam(learning_rate=learning_rate)

    def _calculate_loss(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
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
        predictions = jnp.array(
            self._model(weights=weights, x_data=x_data))
        return self._loss_fun(y_real=y_data, y_pred=predictions)

    @jax.jit
    def _update_step(
            self,
            params: jnp.ndarray,
            opt_state: OptimizerState,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, OptimizerState, jnp.ndarray]:
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
        loss_val, grads = jax.value_and_grad(self._calculate_loss)(params, data, targets)

        # Update quantum model parameters
        updates, opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_val

    @jax.jit
    def _validation_step(
            self,
            params: jnp.ndarray,
            data: jnp.ndarray,
            targets: jnp.ndarray
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

    @abstractmethod
    def optimize(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray,
            y_val: jnp.ndarray,
            verbose: bool
    ):
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
        return NotImplementedError("Subclasses must implement this method.")


@register_pytree_node_class
class QuantumOptimizer(SingleModelOptimizer):
    """
    Quantum Optimizer that extends from a base Optimizer class. This optimizer is specifically designed
    to optimize fully quantum models.

    Class provides functionality for both batch and non-batch training - if batch_size is given
    then batch training is started, otherwise non-batching training runs.

    Args:
        c_params: Parameters of the classical model.
        q_params: Parameters of the quantum model.
        model: Quantum node representing the quantum circuit.
        loss_fn: Loss function for the optimization.
        epochs_num: Number of epochs for the training.
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Size of batches for training. If None, batch processing is not used.
        early_stopping: Instance of EarlyStopping to be used during training.
        retrieve_best_weights: flag indicating if to use BestModelCheckpoint.
    """
    def __init__(
            self,
            c_params: Union[jnp.ndarray, None],
            q_params: Union[jnp.ndarray, None],
            model: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            c_params=c_params,
            q_params=q_params,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )

    def _perform_training_epoch(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            opt_state: OptimizerState
    ) -> Tuple[float, OptimizerState]:
        """
        Performs a single training epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            opt_state: Current state of the optimizer.

        Returns:
            Average training loss for the epoch.
        """

        if self._batch_size:
            total_loss, num_batches = 0.0, 0

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_train, y_train):
                # Update parameters and optimizer states, calculate batch loss
                self._q_params, opt_state, batch_loss = self._update_step(
                    self._q_params, opt_state, x_batch, y_batch)

                # Accumulate total loss and count the batch
                total_loss += batch_loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, perform a single update on the entire dataset
            self._q_params, opt_state, average_loss = self._update_step(
                self._q_params, opt_state, x_train, y_train)

        return average_loss, opt_state

    def _perform_validation_epoch(
            self,
            x_val: jnp.ndarray,
            y_val: jnp.ndarray
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

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_val, y_val):
                # Calculate batch loss
                batch_loss = self._validation_step(self._q_params, x_batch, y_batch)

                # Accumulate total loss and count the batch
                total_loss += batch_loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, calculate loss on the entire dataset
            average_loss = self._validation_step(self._q_params, x_val, y_val)
        return average_loss

    def optimize(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray = None,
            y_val: jnp.ndarray = None,
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

        opt_state = self._opt.init(self._q_params)

        message = ""
        for epoch in range(self._epochs_num):
            train_loss, opt_state = self._perform_training_epoch(
                x_train=x_train, y_train=y_train, opt_state=opt_state)

            if verbose:
                message = f"Epoch {epoch + 1}/{self._epochs_num} - train_loss: {train_loss:.5f}"

            if x_val is not None and y_val is not None:
                val_loss = self._perform_validation_epoch(x_val=x_val, y_val=y_val)

                if self._best_model_checkpoint:
                    self._best_model_checkpoint.update(self._q_params, val_loss)

                if verbose:
                    message += f", val_loss: {val_loss:.5f}"

                # Early stopping logic
                if self._early_stopping:
                    self._early_stopping(val_loss)
                    if self._early_stopping.stop_training:
                        if verbose:
                            print(f"Stopping early at epoch {epoch}.")
                        break

            # Load best model parameters at the end of training
            if self._best_model_checkpoint:
                self._best_model_checkpoint.load_best_model(self)

            if verbose:
                print(message)


@register_pytree_node_class
class ClassicalOptimizer(SingleModelOptimizer):
    def __init__(
            self,

            c_params: Union[jnp.ndarray, None],
            q_params: Union[jnp.ndarray, None],
            model: nn.Module,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            c_params=c_params,
            q_params=q_params,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )

    def _perform_training_epoch(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            opt_state: OptimizerState
    ) -> Tuple[float, OptimizerState]:
        """
        Performs a single training epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            opt_state: Current state of the optimizer.

        Returns:
            Average training loss for the epoch.
        """

        if self._batch_size:
            total_loss, num_batches = 0.0, 0

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_train, y_train):
                # Update parameters and optimizer states, calculate batch loss
                self._c_params, opt_state, batch_loss = self._update_step(
                    self._c_params, opt_state, x_batch, y_batch)

                # Accumulate total loss and count the batch
                total_loss += batch_loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, perform a single update on the entire dataset
            self._c_params, opt_state, average_loss = self._update_step(
                self._c_params, opt_state, x_train, y_train)

        return average_loss, opt_state

    def _perform_validation_epoch(
            self,
            x_val: jnp.ndarray,
            y_val: jnp.ndarray
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

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_val, y_val):
                # Calculate batch loss
                batch_loss = self._validation_step(self._c_params, x_batch, y_batch)

                # Accumulate total loss and count the batch
                total_loss += batch_loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, calculate loss on the entire dataset
            average_loss = self._validation_step(self._c_params, x_val, y_val)
        return average_loss

    def optimize(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray = None,
            y_val: jnp.ndarray = None,
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

        opt_state = self._opt.init(self._c_params)

        message = ""
        for epoch in range(self._epochs_num):
            train_loss, opt_state = self._perform_training_epoch(
                x_train=x_train, y_train=y_train, opt_state=opt_state)

            if verbose:
                message = f"Epoch {epoch + 1}/{self._epochs_num} - train_loss: {train_loss:.5f}"

            if x_val is not None and y_val is not None:
                val_loss = self._perform_validation_epoch(x_val=x_val, y_val=y_val)

                if self._best_model_checkpoint:
                    self._best_model_checkpoint.update(self._c_params, val_loss)

                if verbose:
                    message += f", val_loss: {val_loss:.5f}"

                # Early stopping logic
                if self._early_stopping:
                    self._early_stopping(val_loss)
                    if self._early_stopping.stop_training:
                        if verbose:
                            print(f"Stopping early at epoch {epoch}.")
                        break

            # Load best model parameters at the end of training
            if self._best_model_checkpoint:
                self._best_model_checkpoint.load_best_model(self)

            if verbose:
                print(message)


@register_pytree_node_class
class HybridOptimizer(Optimizer):
    """
    Quantum Optimizer that extends from a base Optimizer class. This optimizer is specifically designed
    to optimize fully quantum models.

    Class provides functionality for both batch and non-batch training - if batch_size is given
    then batch training is started, otherwise non-batching training runs.

    Args:
        c_params: Parameters of the classical model.
        q_params: Parameters of the quantum model.
        model: Quantum node representing the quantum circuit.
        loss_fn: Loss function for the optimization.
        epochs_num: Number of epochs for the training.
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Size of batches for training. If None, batch processing is not used.
        early_stopping: Instance of EarlyStopping to be used during training.
        retrieve_best_weights: flag indicating if to use BestModelCheckpoint.
    """
    def __init__(
            self,
            c_params: Dict[str, Mapping[str, jnp.ndarray]],
            q_params: jnp.ndarray,
            model: qml.qnode,
            loss_fn: Callable,
            epochs_num: int,
            learning_rate: float,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            c_params=c_params,
            q_params=q_params,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs_num=epochs_num,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )
        self._c_opt = optax.adam(learning_rate=learning_rate)
        self._q_opt = optax.adam(learning_rate=learning_rate)

    def _calculate_loss(
            self,
            c_weights: jnp.ndarray,
            q_weights: jnp.ndarray,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
    ):
        """
        Calculates the loss for a given set of weights, input data, and target data.

        Args:
            c_weights: Parameters of the classical model.
            q_weights: Parameters of the quantum model.
            x_data: Input data for the model.
            y_data: Target data for training.

        Returns:
            Computed loss value.
        """
        predictions = jnp.array(
            self._model(
                c_weights=c_weights,
                q_weights=q_weights,
                x_data=x_data)
        )
        return self._loss_fun(y_real=y_data, y_pred=predictions)

    @jax.jit
    def _update_step(
            self,
            c_params: jnp.ndarray,
            q_params: jnp.ndarray,
            c_opt_state: OptimizerState,
            q_opt_state: OptimizerState,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, OptimizerState, OptimizerState, jnp.ndarray]:
        """
        Perform a single update step.

        Args:
            c_params: Classical model parameters.
            q_params: Quantum model parameters.
            c_opt_state: Pytree representing the optimizer state to be updated.
            q_opt_state:
            data: Input data.
            targets: Target data.

        Returns:
            Updated parameters, optimizer state, and loss value.
        """
        loss_val, (c_grads, q_grads) = jax.value_and_grad(
            self._calculate_loss, argnums=(0, 1))(c_params, q_params, data, targets)

        # Classical model parameters update
        c_updates, c_opt_state = self._c_opt.update(c_grads, c_opt_state)
        c_params = optax.apply_updates(c_params, c_updates)

        # Quantum model parameters update
        q_updates, q_opt_state = self._q_opt.update(q_grads, q_opt_state)
        q_params = optax.apply_updates(q_params, q_updates)

        return c_params, q_params, c_opt_state, q_opt_state, loss_val

    @jax.jit
    def _validation_step(
            self,
            c_params: jnp.ndarray,
            q_params: jnp.ndarray,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ) -> np.ndarray:
        """
        Perform a validation step.

        Args:
            c_params: Classical model parameters.
            q_params: Quantum model parameters.
            data: Input data.
            targets: Target data.

        Returns:
            Loss value for the given data and targets.
        """
        return self._calculate_loss(
            c_weights=c_params, q_weights=q_params, x_data=data, y_data=targets
        )

    def _perform_training_epoch(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            c_opt_state: OptimizerState,
            q_opt_state: OptimizerState
    ) -> Tuple[float, OptimizerState, OptimizerState]:
        """
        Performs a single training epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            c_opt_state: Current state of the optimizer.

        Returns:
            Average training loss for the epoch.
        """

        if self._batch_size:
            total_loss, num_batches = 0.0, 0

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_train, y_train):
                # Update parameters and optimizer states, calculate batch loss
                update_args = (
                    self._c_params, self._q_params,  c_opt_state, q_opt_state,  x_batch, y_batch
                )
                self._c_params, self._q_params, c_opt_state, q_opt_state, loss = self._update_step(
                    *update_args)

                # Accumulate total loss and count the batch
                total_loss += loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, perform a single update on the entire dataset
            self._c_params, self._q_params, c_opt_state, q_opt_state, avg_loss = self._update_step(
                self._c_params, self._q_params, c_opt_state, q_opt_state, x_train, y_train)

        return avg_loss, c_opt_state, q_opt_state

    def _perform_validation_epoch(
            self,
            x_val: jnp.ndarray,
            y_val: jnp.ndarray
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

            # Process each batch
            for x_batch, y_batch in self._batch_generator(x_val, y_val):
                batch_loss = self._validation_step(
                    self._c_params, self._q_params, x_batch, y_batch)

                # Accumulate total loss and count the batch
                total_loss += batch_loss
                num_batches += 1

            # Calculate average loss if there are batches processed
            average_loss = total_loss / num_batches if num_batches > 0 else 0
        else:
            # If batching is not used, calculate loss on the entire dataset
            average_loss = self._validation_step(
                self._c_params, self._q_params, x_val, y_val)
        return average_loss

    def optimize(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray = None,
            y_val: jnp.ndarray = None,
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

        c_opt_state = self._c_opt.init(self._c_params)
        q_opt_state = self._q_opt.init(self._q_params)

        message = ""
        for epoch in range(self._epochs_num):
            train_loss, c_opt_state, q_opt_state = self._perform_training_epoch(
                x_train=x_train, y_train=y_train,
                c_opt_state=c_opt_state, q_opt_state=q_opt_state
            )

            if verbose:
                message = f"Epoch {epoch + 1}/{self._epochs_num} - train_loss: {train_loss:.5f}"

            if x_val is not None and y_val is not None:
                val_loss = self._perform_validation_epoch(x_val=x_val, y_val=y_val)

                if self._best_model_checkpoint:
                    self._best_model_checkpoint.update(
                        self._q_params, val_loss, self._c_params)

                if verbose:
                    message += f", val_loss: {val_loss:.5f}"

                # Early stopping logic
                if self._early_stopping:
                    self._early_stopping(val_loss)
                    if self._early_stopping.stop_training:
                        if verbose:
                            print(f"Stopping early at epoch {epoch}.")
                        break

            # Load best model parameters at the end of training
            if self._best_model_checkpoint:
                self._best_model_checkpoint.load_best_model(self)

            if verbose:
                print(message)
