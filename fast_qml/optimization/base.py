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
    Callable, Tuple, Union, Dict, Mapping
)

import optax
import numpy as np
import pennylane as qml
from jax import numpy as jnp

import torch
from torch.utils.data import DataLoader, TensorDataset

from fast_qml.optimization.callbacks import EarlyStopping
from fast_qml.optimization.callbacks import BestModelCheckpoint


class Optimizer:
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
        self._loss_fun = loss_fn
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._early_stopping = early_stopping

        self._best_model_checkpoint = None
        if retrieve_best_weights:
            self._best_model_checkpoint = BestModelCheckpoint()

        self._opt = optax.adam(learning_rate=learning_rate)

    def tree_flatten(self):
        children = [self._c_params, self._q_params, self._batch_stats]
        aux_data = {
            'model': self._model,
            'loss_fn': self._loss_fun,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @property
    def weights(
            self
    ) -> Union[jnp.ndarray, Dict[str, Mapping[str, jnp.ndarray]], Tuple]:
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

    def _set_dataloaders(
            self,
            train_data: Union[np.ndarray, DataLoader],
            train_targets: Union[np.ndarray, None],
            val_data: Union[np.ndarray, DataLoader],
            val_targets: Union[np.ndarray, None]
    ) -> Tuple[DataLoader, DataLoader]:

        # Convert raw arrays to DataLoader instances if necessary
        if not isinstance(train_data, DataLoader):
            train_dataset = TensorDataset(
                torch.from_numpy(train_data).float(),
                torch.from_numpy(train_targets).float()
            )
            train_loader = DataLoader(train_dataset, batch_size=self._batch_size)
        else:
            train_loader = train_data

        if not isinstance(val_data, DataLoader):
            val_dataset = TensorDataset(
                torch.from_numpy(val_data).float(),
                torch.from_numpy(val_targets).float()
            )
            val_loader = DataLoader(val_dataset, batch_size=self._batch_size)
        else:
            val_loader = val_data

        return train_loader, val_loader


    @abstractmethod
    def optimize(
            self,
            train_data: Union[jnp.ndarray, DataLoader],
            train_targets: Union[jnp.ndarray, None],
            val_data: Union[jnp.ndarray, DataLoader],
            val_targets: Union[jnp.ndarray, None],
            epochs_num: int,
            verbose: bool
    ):
        return NotImplementedError("Subclasses must implement this method.")

