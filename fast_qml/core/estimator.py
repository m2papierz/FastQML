# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import os
import dataclasses
from abc import abstractmethod
from pathlib import Path

from typing import Callable
from typing import Union
from typing import Dict
from typing import Mapping
from typing import Any

import jax
import torch
import pickle
import numpy as np
from jax import numpy as jnp
from torch.utils.data import DataLoader

from fast_qml.core.callbacks import EarlyStopping
from fast_qml.core.optimizer import QuantumOptimizer
from fast_qml.core.optimizer import ClassicalOptimizer
from fast_qml.core.optimizer import HybridOptimizer


@dataclasses.dataclass
class EstimatorParameters:
    """
    A dataclass to hold parameters for an estimator.

    Attributes:
        c_weights: classical model weights
        q_weights: quantum weights
        batch_stats: batch statistics for classical model
    """
    c_weights: Union[jnp.ndarray, Dict[str, Any]] = None
    q_weights: jnp.ndarray = None
    batch_stats: Union[jnp.ndarray, Dict[str, Any]] = None

    def __post_init__(self):
        if self.c_weights is not None and not isinstance(self.c_weights, (jnp.ndarray, dict)):
            raise TypeError("c_weights must be either jnp.ndarray or dict")
        if self.q_weights is not None and not isinstance(self.q_weights, jnp.ndarray):
            raise TypeError("q_weights must be jnp.ndarray")
        if self.batch_stats is not None and not isinstance(self.batch_stats, (jnp.ndarray, dict)):
            raise TypeError("batch_stats must be either jnp.ndarray or dict")

    @property
    def params_num(self) -> int:
        total_params = 0
        if self.c_weights is not None:
            total_params += sum(x.size for x in jax.tree_leaves(self.c_weights))
        if self.q_weights is not None:
            total_params += len(self.q_weights)
        return total_params


class Estimator:
    """
    An abstract base class for creating machine learning estimators. This class provides a template
    for  implementing machine learning estimators with basic functionalities of model training, saving,
    and loading.
    """
    random_seed: int = 0

    def __init__(
            self,
            loss_fn: Callable,
            optimizer_fn: Callable,
            estimator_type: str,
            init_args: Dict[str, Any]
    ):
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self._init_args = init_args
        self.input_shape = None

        # Initiate parameters
        self.init_parameters()

        # Initiate estimator optimizer
        self._trainer = self._init_trainer(estimator_type)

    @staticmethod
    def _init_trainer(estimator_type: str):
        """Initializes and returns an optimizer based on the specified estimator type.

        Args:
            estimator_type: The type of optimizer to initialize. Valid options are
            'quantum', 'classical', and 'hybrid'.

        Returns:
            An instance of optimizer based on the estimator type.
        """
        if estimator_type == 'quantum':
            return QuantumOptimizer
        elif estimator_type == 'classical':
            return ClassicalOptimizer
        elif estimator_type == 'hybrid':
            return HybridOptimizer
        else:
            raise ValueError(
                f"Invalid optimizer type: {estimator_type},"
                f" available options are {'quantum', 'classical', 'hybrid'}"
            )

    def init_parameters(self):
        """
        Initiates estimator model parameters.
        """
        # As implements an explicit PRNG, we need to artificially change random seed, in order to
        # achiever pseudo sampling, allowing to get different numbers each time we sample parameters
        self.random_seed += 1

        # Initiate estimator parameters with sampled numbers
        self.params = EstimatorParameters(
            **self._sample_parameters(**self._init_args)
        )

    @abstractmethod
    def _sample_parameters(self, **kwargs):
        """
        Abstract method for sampling estimator parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def model(
            self,
            x_data: jnp.ndarray,
            q_weights: Union[jnp.ndarray, None] = None,
            c_weights: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            training: Union[bool, None] = None,
            q_model_probs: Union[bool] = False
    ):
        """
        Abstract method for defining estimator model.
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
        ....

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
        trainer = self._trainer(
            c_params=self.params.c_weights,
            q_params=self.params.q_weights,
            batch_stats=self.params.batch_stats,
            model=self.model,
            loss_fn=self.loss_fn,
            c_optimizer=self.optimizer_fn(learning_rate),
            q_optimizer=self.optimizer_fn(learning_rate),
            batch_size=batch_size,
            early_stopping=early_stopping
        )

        trainer.optimize(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            epochs_num=num_epochs,
            verbose=verbose
        )

        self.params = EstimatorParameters(**trainer.parameters)

    def model_save(
            self,
            directory: str,
            name: str
    ) -> None:
        """
        Saves the model parameters to a pickle file. This method saves the current state of the model
        parameters to a specified directory with a given name.

        Args:
            directory: The directory path where the model should be saved.
            name: The name of the file to save the model parameters.

        The model is saved in a binary file with a `.model` extension.
        """
        dir_ = Path(directory)
        if not os.path.exists(dir_):
            os.mkdir(dir_)

        with open(dir_ / f"{name}.model", 'wb') as f:
            pickle.dump(self.params, f)

    def model_load(
            self,
            path: str
    ) -> None:
        """
        Loads model parameters from a pickle file. This method loads the model parameters from a specified
        file path, updating the `params` attribute of the instance.

        Args:
            path: The file path to load the model parameters from.

        The method expects a binary file with saved model parameters.
        """
        with open(Path(path), 'rb') as f:
            self.params = pickle.load(f)
