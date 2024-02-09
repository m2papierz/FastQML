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
from typing import  Callable, Tuple, Union, Dict, Mapping

import jax
import optax
import torch
import numpy as np
import pennylane as qml
from jax import numpy as jnp

from torch.utils.data import DataLoader, TensorDataset
from jax.example_libraries.optimizers import OptimizerState
from jax.tree_util import register_pytree_node_class

from fast_qml.core.callbacks import EarlyStopping
from fast_qml.core.callbacks import BestModelCheckpoint


class Optimizer:
    """
    Base class for optimizers used in quantum machine learning models.

    This class provides the foundational structure for implementing various core
    algorithms. It is designed to integrate with both classical Flax deep learning models
    and quantum models.

    Args:
        c_params: Parameters of the classical model.
        q_params: Parameters of the quantum model.
        model: Quantum node representing the quantum circuit.
        loss_fn: Loss function used for core.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        early_stopping: Instance of EarlyStopping to be used during training.
        retrieve_best_weights: Boolean flag indicating if to use BestModelCheckpoint.
    """
    def __init__(
            self,
            c_params: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            q_params: Union[jnp.ndarray, None],
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            model: [qml.qnode, Callable],
            loss_fn: Callable,
            learning_rate: float,
            batch_size: int,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        self._c_params = c_params
        self._q_params = q_params
        self._batch_stats = batch_stats
        self._model = model
        self._loss_fn = loss_fn
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._early_stopping = early_stopping

        self._best_model_checkpoint = None
        if retrieve_best_weights:
            self._best_model_checkpoint = BestModelCheckpoint()

        self._opt = optax.adam(learning_rate=learning_rate)
        self._train_loader, self._val_loader = None, None

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations. This method is used
        for JAX automatic differentiation and is required for the class to work
        with jax.jit optimizations.
        """
        children = [self._c_params, self._q_params, self._batch_stats]
        aux_data = {
            'model': self._model,
            'loss_fn': self._loss_fn,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size
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

    def _create_dataloader(
            self,
            data: Union[np.ndarray, torch.Tensor],
            targets: Union[np.ndarray, torch.Tensor],
    ) -> DataLoader:
        """
        Creates a DataLoader from tensors or numpy arrays.
        """
        if isinstance(data, np.ndarray):
            data, targets = torch.from_numpy(data), torch.from_numpy(targets)
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=self._batch_size)

    @staticmethod
    def _validate_data_targets(
            data: Union[np.ndarray, torch.Tensor, DataLoader],
            targets: Union[np.ndarray, torch.Tensor, DataLoader]
    ) -> None:
        """
        Validates that targets are provided for numpy arrays and tensors.
        """
        if not isinstance(data, (DataLoader, np.ndarray, torch.Tensor)):
            raise TypeError(
                "Data must be a DataLoader, numpy array, or torch.Tensor."
            )

        if not isinstance(data, DataLoader) and targets is None:
            raise ValueError(
                "Targets must be provided for numpy arrays or tensors."
            )

    def _set_dataloaders(
            self,
            train_data: Union[np.ndarray, torch.Tensor, DataLoader],
            train_targets: Union[np.ndarray, torch.Tensor, None],
            val_data: Union[np.ndarray, torch.Tensor, DataLoader],
            val_targets: Union[np.ndarray, torch.Tensor, None]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Configures data loaders for training and validation data.

        Converts training and validation data into DataLoader instances if they are not already DataLoader
        instances. For NumPy arrays and PyTorch tensors, wraps the data and targets into a TensorDataset, which
        is then used to create a DataLoader with batch size defined in `self._batch_size`.

        Args:
            train_data: The training dataset, which can be a DataLoader, a NumPy array, or a PyTorch tensor.
            train_targets: The targets corresponding to `train_data`, required if `train_data` is not a DataLoader.
            val_data: The validation dataset, which can be a DataLoader, a NumPy array, or a PyTorch tensor.
            val_targets: The targets corresponding to `val_data`, required if `val_data` is not a DataLoader.

        Returns:
            A tuple containing two DataLoader instances for the training and validation datasets.

        Raises:
            ValueError: If targets are not provided for non-DataLoader datasets.
            TypeError: If data inputs are not DataLoader, NumPy array, or PyTorch tensor.
        """
        self._validate_data_targets(train_data, train_targets)
        self._validate_data_targets(val_data, val_targets)

        if not isinstance(train_data, DataLoader):
            train_loader = self._create_dataloader(
                data=train_data, targets=train_targets)
        else:
            train_loader = train_data

        if not isinstance(val_data, DataLoader):
            val_loader = self._create_dataloader(
                data=val_data, targets=val_targets)
        else:
            val_loader = val_data

        return train_loader, val_loader

    @abstractmethod
    def optimize(
            self,
            train_data: Union[np.ndarray, torch.Tensor, DataLoader],
            train_targets: Union[np.ndarray, torch.Tensor, None],
            val_data: Union[np.ndarray, torch.Tensor, DataLoader],
            val_targets: Union[np.ndarray, torch.Tensor, None],
            epochs_num: int,
            verbose: bool
    ):
        return NotImplementedError("Subclasses must implement this method.")


@register_pytree_node_class
class QuantumOptimizer(Optimizer):
    def __init__(
            self,
            c_params: Union[jnp.ndarray, None],
            q_params: Union[jnp.ndarray, None],
            batch_stats: Union[jnp.ndarray, None],
            model: [qml.qnode, Callable],
            loss_fn: Callable,
            learning_rate: float,
            batch_size: int,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        super().__init__(
            c_params=c_params,
            q_params=q_params,
            batch_stats=batch_stats,
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping=early_stopping,
            retrieve_best_weights=retrieve_best_weights
        )

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
        predictions = self._model(weights=weights, x_data=x_data)
        predictions = jnp.array(predictions).T
        loss_val = self._loss_fn(predictions, y_data).mean()
        return loss_val

    @jax.jit
    def _train_step(
            self,
            params: jnp.ndarray,
            opt_state: OptimizerState,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ):
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
    ) -> jnp.ndarray:
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


    def _training_epoch(
            self,
            opt_state: OptimizerState
    ) -> Tuple[float, OptimizerState]:
        """
        Performs a single training epoch.

        Args:
            opt_state: Current state of the optimizer.

        Returns:
            Average training loss for the epoch.
        """
        total_loss, num_batches = 0.0, 0

        # Process each batch
        for x_batch, y_batch in iter(self._train_loader):
            # Update parameters and optimizer states, calculate batch loss
            x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
            self._q_params, opt_state, batch_loss = self._train_step(
                self._q_params, opt_state, x_batch, y_batch)

            # Accumulate total loss and count the batch
            total_loss += batch_loss
            num_batches += 1

        # Calculate average loss if there are batches processed
        average_loss = total_loss / num_batches if num_batches > 0 else 0

        return average_loss, opt_state

    def _validation_epoch(self) -> float:
        """
        Performs a validation step.

        Returns:
            Average validation loss.
        """
        total_loss, num_batches = 0.0, 0

        # Process each batch
        for x_batch, y_batch in iter(self._val_loader):
            # Calculate batch loss
            x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
            batch_loss = self._validation_step(self._q_params, x_batch, y_batch)

            # Accumulate total loss and count the batch
            total_loss += batch_loss
            num_batches += 1

        # Calculate average loss if there are batches processed
        average_loss = total_loss / num_batches if num_batches > 0 else 0

        return average_loss


    def optimize(
            self,
            train_data: Union[np.ndarray, torch.Tensor, DataLoader],
            train_targets: Union[np.ndarray, torch.Tensor, None],
            val_data: Union[np.ndarray, torch.Tensor, DataLoader],
            val_targets: Union[np.ndarray, torch.Tensor, None],
            epochs_num: int,
            verbose: bool
    ) -> None:

        self._train_loader, self._val_loader = self._set_dataloaders(
            train_data=train_data, train_targets=train_targets,
            val_data=val_data, val_targets=val_targets
        )

        opt_state = self._opt.init(self._q_params)

        for epoch in range(epochs_num):
            train_loss, opt_state = self._training_epoch(opt_state=opt_state)
            val_loss = self._validation_epoch()

            if self._best_model_checkpoint:
                self._best_model_checkpoint.update(self._q_params, val_loss)

            # Early stopping logic
            if self._early_stopping:
                self._early_stopping(val_loss)
                if self._early_stopping.stop_training:
                    if verbose:
                        print(f"Stopping early at epoch {epoch}.")
                    break

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs_num} - "
                    f"train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}"
                )

        # Load best model parameters at the end of training
        if self._best_model_checkpoint:
            self._best_model_checkpoint.load_best_model(self)
