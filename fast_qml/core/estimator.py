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
import pennylane as qml
from jax import vmap
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


class Estimator:
    """
    An abstract base class for creating machine learning estimators. This class provides a template
    for  implementing machine learning estimators with basic functionalities of model training, saving,
    and loading.
    """
    def __init__(
            self,
            loss_fn: Callable,
            optimizer_fn: Callable,
            estimator_type: str
    ):
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        self.params = EstimatorParameters()
        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)

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

    def _compute_fisher_matrix(
            self,
            x: jnp.ndarray,
            q_params,
            c_params,
            batch_stats
    ) -> jnp.ndarray:
        """
        Computes the Fisher Information Matrix for a given input.

        This method computes the Fisher Information Matrix (FIM) for the model parameters based on the input.
        The FIM is a way of estimating the amount of information that an observable random variable carries
        about an unknown parameter upon which the probability of the random variable depends.

        Args:
            x: The input data for which the Fisher Information Matrix is to be computed.

        Returns:
            The Fisher Information Matrix, a square array of shape (n_params, n_params), where
            `n_params` is the number of model outputs (observables).
        """
        # Compute model output probabilities
        proba = self.model(
            x_data=x, q_weights=q_params,
            c_weights=c_params, batch_stats=batch_stats,
            training=False, q_model_probs=True)

        # Compute derivatives of probabilities in regard to model parameters
        proba_d = jax.jacfwd(
            self.model)(x, q_params, c_params, batch_stats, False, True)

        # Exclude zero values and calculate 1 / proba
        non_zeros_proba = qml.math.where(
            proba > 0, proba, qml.math.ones_like(proba))
        one_over_proba = qml.math.where(
            proba > 0, qml.math.ones_like(proba), qml.math.zeros_like(proba))
        one_over_proba = one_over_proba / non_zeros_proba

        # Cast, reshape, and transpose matrix to get (n_params, n_params) array
        proba_d = qml.math.cast_like(proba_d, proba)
        proba_d = qml.math.reshape(proba_d, (len(proba), -1))
        proba_d_over_p = qml.math.transpose(proba_d) * one_over_proba

        return proba_d_over_p @ proba_d

    def fisher_information(
            self,
            x_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the normalized Fisher Information Matrix (FIM) averaged over a batch of data.

        Args:
            x_data: The input data, a batch of observations for which the Fisher Information Matrix is
            to be computed.

        Returns:
            The normalized Fisher Information Matrix, averaged over the input batch of data.
        """
        # Unpack model parameters
        c_params, q_params, batch_stats = dataclasses.asdict(self.params).values()

        # Create batched version of fisher matrix computation
        _compute_fisher_matrix_batched = vmap(
            self._compute_fisher_matrix, in_axes=(0, None, None, None))

        # Compute FIM average over the given data
        fim = jnp.mean(_compute_fisher_matrix_batched(
            x_data, q_params, c_params, batch_stats), axis=0)
        fisher_inf_norm = (fim - np.min(fim)) / (np.max(fim) - np.min(fim))

        return fisher_inf_norm


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
