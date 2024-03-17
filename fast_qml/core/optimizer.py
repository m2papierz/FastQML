# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from collections import OrderedDict

from typing import Callable
from typing import Tuple
from typing import Union

import jax
import optax
import torch
import numpy as np
from jax import numpy as jnp

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from jax.example_libraries.optimizers import OptimizerState
from jax.tree_util import register_pytree_node_class

from fast_qml.core.callbacks import EarlyStopping
from fast_qml.core.callbacks import BestModelCheckpoint


@register_pytree_node_class
class ParametersOptimizer:
    def __init__(
            self,
            parameters: OrderedDict,
            forward_fn: Callable,
            loss_fn: Callable,
            optimizer: optax.GradientTransformation,
            batch_size: int,
            early_stopping: EarlyStopping = None,
            retrieve_best_weights: bool = True
    ):
        self._parameters = parameters
        self._forward_fn = forward_fn
        self._loss_fn = loss_fn
        self._opt = optimizer
        self._batch_size = batch_size

        self._early_stopping = early_stopping

        self._best_model_checkpoint = None
        if retrieve_best_weights:
            self._best_model_checkpoint = BestModelCheckpoint()

        self._train_loader, self._val_loader = None, None

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        children = [self._parameters]
        aux_data = {
            'forward_fn': self._forward_fn,
            'loss_fn': self._loss_fn,
            'optimizer': self._opt,
            'batch_size': self._batch_size
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

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
        instances. For NumPy arrays and PyTorch tensors, wraps the data and targets into a TensorDataset,
        which is then used to create a DataLoader with batch size defined in `self._batch_size`.

        Args:
            train_data: Training dataset, which can be a DataLoader/NumPy array/PyTorch tensor.
            train_targets: Targets corresponding to `train_data`, required if `train_data` is not a DataLoader.
            val_data: Validation dataset, which can be a DataLoader/NumPy array/PyTorch tensor
            val_targets: Targets corresponding to `val_data`, required if `val_data` is not a DataLoader.

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

    @jax.jit
    def _compute_loss(
            self,
            params: OrderedDict,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
    ) -> float:
        """
        Computes the loss of the estimator for a given batch of data.

        Args:
            params: Parameters of the estimator.
            x_data: Input data for the model.
            y_data: Target data for training.

        Returns:
            Computed loss value.
        """
        predictions = self._forward_fn(
            x_data=x_data,
            parameters=params,
            return_q_probs=False,
            flatten_c_output=False
        )

        return self._loss_fn(predictions, y_data).mean()

    @jax.jit
    def _train_step(
            self,
            params: OrderedDict,
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
            self._compute_loss, argnums=0
        )(params=params, x_data=data, y_data=targets)

        # Update quantum model parameters
        updates, opt_state = self._opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_val

    @jax.jit
    def _validation_step(
            self,
            params: jnp.ndarray,
            data: jnp.ndarray,
            targets: jnp.ndarray
    ) -> float:
        """
        Perform a validation step.

        Args:
            params: Model parameters.
            data: Input data.
            targets: Target data.

        Returns:
            Loss value for the given data and targets.
        """
        return self._compute_loss(
            params=params, x_data=data, y_data=targets
        )

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
            self._parameters, opt_state, batch_loss = self._train_step(
                self._parameters, opt_state, x_batch, y_batch)

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
            batch_loss = self._validation_step(
                self._parameters, x_batch, y_batch)

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
        """
        Executes the optimization loop for training and validating the quantum model.

        Args:
            train_data: The training dataset. This can be in the form of a DataLoader, which directly
                provides batches of data, or a NumPy array or PyTorch tensor, from which a DataLoader will be created.
            train_targets: The target labels or values for the training data. Required if `train_data` is not
                a DataLoader. If `train_data` is a DataLoader, this should be None.
            val_data: The validation dataset. Similar to `train_data`, this can be a DataLoader, a NumPy
                array, or a PyTorch tensor.
            val_targets: The target labels or values for the validation data. Required if `val_data` is not
                a DataLoader. If `val_data` is a DataLoader, this should be None.
            epochs_num: The number of epochs to run the optimization loop.
            verbose: Boolean flag indicating whether to print training progress.
        """
        self._train_loader, self._val_loader = self._set_dataloaders(
            train_data=train_data, train_targets=train_targets,
            val_data=val_data, val_targets=val_targets
        )

        self._opt = optax.chain(optax.clip(1.0), self._opt)
        opt_state = self._opt.init(self._parameters)

        for epoch in range(epochs_num):
            train_loss, opt_state = self._training_epoch(opt_state=opt_state)
            val_loss = self._validation_epoch()

            # if self._best_model_checkpoint:
            #     # TODO: Update model checkpoint
            #     self._best_model_checkpoint.update(
            #         current_val_loss=val_loss, current_q_params=self._q_params)
            #
            # # Early stopping logic
            # if self._early_stopping:
            #     self._early_stopping(val_loss)
            #     if self._early_stopping.stop_training:
            #         if verbose:
            #             print(f"Stopping early at epoch {epoch}.")
            #         break

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs_num} - "
                    f"train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}"
                )

        # # Load best model parameters at the end of training
        # if self._best_model_checkpoint:
        #     self._best_model_checkpoint.load_best_model(self)
