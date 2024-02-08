# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.



from typing import (
    Callable, Tuple, Union, Dict, Mapping
)

import jax
import optax
import numpy as np
import pennylane as qml
from jax import numpy as jnp
from torch.utils.data import DataLoader
from jax.example_libraries.optimizers import OptimizerState
from jax.tree_util import register_pytree_node_class

from fast_qml.optimization.base import Optimizer
from fast_qml.optimization.callbacks import EarlyStopping


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
        self._train_loader, self._val_loader = None, None

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


    def _perform_training_epoch(
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
            x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)

            # Update parameters and optimizer states, calculate batch loss
            self._q_params, opt_state, batch_loss = self._update_step(
                self._q_params, opt_state, x_batch, y_batch)

            # Accumulate total loss and count the batch
            total_loss += batch_loss
            num_batches += 1

        # Calculate average loss if there are batches processed
        average_loss = total_loss / num_batches if num_batches > 0 else 0

        return average_loss, opt_state

    def _perform_validation_epoch(self) -> float:
        """
        Performs a validation step.

        Returns:
            Average validation loss.
        """
        total_loss, num_batches = 0.0, 0

        # Process each batch
        for x_batch, y_batch in iter(self._val_loader):
            x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
            # Calculate batch loss
            batch_loss = self._validation_step(self._q_params, x_batch, y_batch)

            # Accumulate total loss and count the batch
            total_loss += batch_loss
            num_batches += 1

        # Calculate average loss if there are batches processed
        average_loss = total_loss / num_batches if num_batches > 0 else 0

        return average_loss


    def optimize(
            self,
            train_data: Union[np.ndarray, DataLoader],
            train_targets: Union[np.ndarray, None],
            val_data: Union[np.ndarray, DataLoader],
            val_targets: Union[np.ndarray, None],
            epochs_num: int,
            verbose: bool
    ) -> None:

        self._train_loader, self._val_loader = self._set_dataloaders(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets
        )

        opt_state = self._opt.init(self._q_params)

        for epoch in range(epochs_num):
            train_loss, opt_state = self._perform_training_epoch(
                opt_state=opt_state)
            val_loss = self._perform_validation_epoch()

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
