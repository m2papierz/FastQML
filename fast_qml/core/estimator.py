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
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Union, List, Any, Tuple, Dict, Mapping

import jax
import torch
import pickle
import flax.linen as nn
import numpy as np
import pennylane as qml
from jax import numpy as jnp
from torch.utils.data import DataLoader

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm
from fast_qml.core.callbacks import EarlyStopping

from fast_qml.core.optimizer import (
    QuantumOptimizer, ClassicalOptimizer, HybridOptimizer)


class EstimatorParameters:
    """
    Initializes parameters for various types of models, including classical, quantum, and hybrid models.

    This class provides a flexible way to initialize parameters for different types of models
    by specifying the model type and related configuration. It supports vectorized quantum ansatz (VQA),
    quantum neural networks (QNN), classical neural networks, and hybrid models combining classical
    and quantum components.

    Attributes:
        _parameters: A dictionary to store the initialized parameters.
        _inp_rng: A JAX PRNG key for input-related randomness.
        _init_rng: A JAX PRNG key for initialization-related randomness.
    """
    def __init__(self, seed: int = 42):
        self._parameters: Dict[str, Union[jnp.ndarray, Any]] = {}
        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=seed), num=2)

    def add_parameters(
            self,
            name: str,
            array: Union[jnp.ndarray, Any]
    ) -> None:
        """
        Adds a parameter to the internal dictionary of parameters.

        Args:
            name: The name of the parameters.
            array: The parameters array.
        """
        self._parameters[name] = array

    def _init_vqa_params(
            self,
            n_ansatz_params: Union[int, List[int]]
    ) -> None:
        """
        Initializes parameters for a variational quantum model.

        Args:
            n_ansatz_params: The number of parameters in the ansatz.
        """
        if isinstance(n_ansatz_params, int):
            shape = [n_ansatz_params]
        else:
            shape = [*n_ansatz_params]

        weights = 0.1 * jax.random.normal(self._init_rng, shape=shape)
        self.add_parameters(name="q_weights", array=weights)

    def _init_qnn_params(
            self,
            n_ansatz_params: Union[int, List[int]],
            layers_n: int
    ) -> None:
        """
         Initializes parameters for a quantum neural network model.

         Args:
             n_ansatz_params: The number of parameters in the ansatz.
             layers_n: The number of layers in the model.
        """
        if isinstance(n_ansatz_params, int):
            shape = (layers_n, n_ansatz_params)
        else:
            shape = (layers_n, *n_ansatz_params)

        weights = 0.1 * jax.random.normal(self._init_rng, shape=shape)
        self.add_parameters(name="q_weights", array=weights)

    def _init_classical_params(
            self,
            c_model: nn.Module,
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ) -> None:
        """
        Initializes parameters for a classical neural network model.

        Args:
            c_model: The classical neural network model for which parameters are initialized.
            input_shape: The shape of the input to the model.
            batch_norm: Indicates whether batch normalization is used.
        """
        if isinstance(input_shape, int):
            shape = (1, input_shape)
        else:
            shape = (1, *input_shape)

        c_inp = jax.random.normal(self._inp_rng, shape=shape)

        if batch_norm:
            variables = c_model.init(self._init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']

            self.add_parameters(name="c_weights", array=weights)
            self.add_parameters(name="batch_stats", array=batch_stats)
        else:
            variables = c_model.init(self._init_rng, c_inp)
            weights = variables['params']

            self.add_parameters(name="c_weights", array=weights)

    def _init_hybrid_params(
            self,
            c_model: nn.Module,
            q_model_params: Union[int, Tuple[int]],
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ) -> None:
        """
        Initializes parameters for a hybrid quantum-classical model.

        Args:
            c_model: The classical neural network model for which parameters are initialized.
            q_model_params: The parameters for the quantum component of the hybrid model.
            input_shape: The shape of the input to the model.
            batch_norm: Indicates whether batch normalization is used in classical model.
        """
        self._init_classical_params(
            c_model=c_model,
            input_shape=input_shape,
            batch_norm=batch_norm
        )
        self.add_parameters(name="q_weights", array=q_model_params)

    def __call__(
            self,
            estimator_type: str,
            n_ansatz_params: Union[int, List[int], None] = None,
            layers_n: Union[int, None] = None,
            c_model: Union[nn.Module, None] = None,
            q_model_params: Union[int, Tuple[int], None] = None,
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ):
        """
        Initializes parameters based on the specified estimator type and configuration.

        Args:
            estimator_type: The type of estimator/model for which parameters are being initialized.
                Can be 'vqa', 'qnn', 'classical', or 'hybrid'.
            n_ansatz_params: The number or shape of parameters in quantum ansatz.
            layers_n: The number of layers for quantum neural networks.
            c_model: The classical model component of the hybrid model.
            q_model_params: The parameters for the quantum component of the hybrid model.
            input_shape: The input shape for the classical model.
            batch_norm: Indicates whether batch normalization is used in the classical model.

        Returns:
            A dictionary containing the initialized parameters.

        Raises:
            ValueError: If an unknown estimator type is provided.
        """
        if estimator_type == 'vqa':
            self._init_vqa_params(
                n_ansatz_params=n_ansatz_params)
        elif estimator_type == 'qnn':
            self._init_qnn_params(
                n_ansatz_params=n_ansatz_params, layers_n=layers_n)
        elif estimator_type == 'classical':
            self._init_classical_params(
                c_model=c_model, input_shape=input_shape, batch_norm=batch_norm)
        elif estimator_type == 'hybrid':
            self._init_hybrid_params(
                c_model=c_model, q_model_params=q_model_params,
                input_shape=input_shape, batch_norm=batch_norm)
        else:
            raise ValueError(
                f"Unknown estimator type: {estimator_type}. "
                f"Available types are {['vqa', 'qnn', 'classical', 'hybrid']}"
            )

        return self._parameters


class Estimator:
    """
    An abstract base class for creating machine learning estimators.

    This class provides a template for implementing machine learning estimators with basic functionalities
    for model saving, and loading. It requires subclasses to implement the parameter initialization method.
    """
    def __init__(self):
        self.params = None
        self._params_initializer = EstimatorParameters()

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


class QuantumEstimator(Estimator):
    """
    Provides a framework for implementing quantum machine learning estimators.

    This base class is designed for the creation of quantum estimators, facilitating the integration of quantum
    circuits with machine learning algorithms. It is intended to be subclassed for specific quantum model
    implementations, where the quantum circuit is defined by a feature map, an ansatz, and a measurement operation.

    Args:
        n_qubits: The number of qubits in the quantum circuit.
        feature_map: The feature map for encoding classical data into quantum states.
        ansatz: The variational form for the quantum circuit.
        measurement_op: The measurement operator or observable used to measure the quantum state.
        loss_fn: The loss function used to evaluate the model's predictions against the true outcomes.
        optimizer: The optimization algorithm.
        measurements_num: The number of wires on which to perform measurements.

    """
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            loss_fn: Callable,
            optimizer: Callable,
            measurement_op: Callable = qml.PauliZ,
            measurements_num: int = 1
    ):
        super().__init__()

        # Validate measurement operation
        if not self._is_valid_measurement_op(measurement_op):
            raise ValueError("Invalid measurement operation provided.")

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num

        self._device = qml.device(
            name="default.qubit.jax", wires=self._n_qubits)


    @staticmethod
    def _is_valid_measurement_op(measurement_op):
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

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
            q_params=self.params,
            batch_stats=None,
            model=self.q_model,
            loss_fn=self.loss_fn,
            c_optimizer=None,
            q_optimizer=self.optimizer(learning_rate),
            batch_size=batch_size,
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

        self.params = optimizer.parameters


class ClassicalEstimator(Estimator):
    """
    Base class for creating classical machine learning estimators based on Flax models.

    This class provides a structured framework for defining and training classical machine learning models,
    particularly those based on neural networks. It should be subclassed to implement specific model
    architectures and functionalities.

    Args:
        c_model: The classical Flax neural network to be trained.
        loss_fn: The loss function used to evaluate the model's performance.
        optimizer: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
    """
    def __init__(
            self,
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__()

        self._c_model = c_model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self.batch_norm = batch_norm

        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)

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
        Trains the classical neural network estimator on the provided dataset.

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
        if self.batch_norm:
            weights, batch_stats = (
                self.params['c_weights'], self.params['batch_stats'])
        else:
            weights, batch_stats = self.params['c_weights'], None

        optimizer = ClassicalOptimizer(
            c_params=weights,
            q_params=None,
            batch_stats=batch_stats,
            model=self._model,
            loss_fn=self._loss_fn,
            c_optimizer=self._optimizer(learning_rate),
            q_optimizer=None,
            batch_size=batch_size,
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

        self.params['c_weights'] = optimizer.parameters
        self.params['batch_stats'] = optimizer.batch_stats


class HybridEstimator(Estimator):
    """
    Provides a framework for creating hybrid quantum-classical machine learning estimators.

    This base class facilitates the integration of classical neural network models with quantum circuits to form
    hybrid models. It enables the use of quantum circuits as components within a larger classical model architecture
    or vice versa, aiming to leverage the strengths of both quantum and classical approaches.

    Args:
        c_model: The classical model component.
        q_model: The quantum model component, defined as an instance of a QuantumEstimator subclass.
    """
    def __init__(
            self,
            c_model: nn.Module,
            q_model: Any
    ):
        super().__init__()

        self._c_model = c_model
        self._q_model = q_model
        self._loss_fn = q_model.loss_fn
        self._optimizer = q_model.optimizer
        self._batch_norm = c_model.batch_norm

        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)

    @abstractmethod
    def _model(
            self,
            c_weights: Dict[str, Mapping[str, jnp.ndarray]],
            q_weights: jnp.ndarray,
            x_data: jnp.ndarray,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            training: bool
    ):
        """
        Defines the hybrid model by combining classical and quantum models.
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
        Trains the hybrid quantum-classical estimator on the provided dataset.

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
        if self._batch_norm:
            c_weights, batch_stats = (
                self.params['c_weights'], self.params['batch_stats'])
        else:
            c_weights, batch_stats = self.params['c_weights'], None
        q_weights = self.params['q_weights']

        optimizer = HybridOptimizer(
            c_params=c_weights,
            q_params=q_weights,
            batch_stats=batch_stats,
            model=self._model,
            loss_fn=self._q_model.loss_fn,
            c_optimizer=self._optimizer(learning_rate),
            q_optimizer=self._optimizer(learning_rate),
            batch_size=batch_size,
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

        self.params['c_weights'], self.params['q_weights'] = optimizer.parameters
        self.params['batch_stats'] = optimizer.batch_stats
