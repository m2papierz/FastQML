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

from typing import Callable
from typing import Union
from typing import Dict
from typing import Mapping
from typing import Any
from typing import List
from typing import Tuple

import torch
import numpy as np
import pennylane as qml
from torch.utils.data import DataLoader

import jax
import flax.linen as nn
from jax import numpy as jnp

from fast_qml.quantum_circuits.data_encoding import FeatureMap
from fast_qml.quantum_circuits.data_encoding import AmplitudeEmbedding
from fast_qml.quantum_circuits.ansatz import VariationalForm

from fast_qml.core.dataclasses import EstimatorLayerParameters
from fast_qml.core.dataclasses import EstimatorParameters
from fast_qml.core.callbacks import EarlyStopping
from fast_qml.core.optimizer import QuantumOptimizer
from fast_qml.core.optimizer import ClassicalOptimizer
from fast_qml.core.optimizer import HybridOptimizer


class EstimatorLayer:
    def __init__(
            self,
            init_args: Dict[str, Any]
    ):
        self._random_seed = 0
        self._init_args = init_args

        # Initiate layer parameters
        self.params = EstimatorLayerParameters(
            q_params=None, c_params=None, batch_stats=None)
        self.init_parameters()

    def init_parameters(self):
        """
        Initiates estimator layer parameters.
        """
        # To ensure varied outcomes in pseudo-random sampling with JAX's explicit PRNG system,
        # we need to manually increment the random seed for each sampling
        self._random_seed += 1

        # Initiate layer parameters with sampled parameters
        self.params = EstimatorLayerParameters(
            **self._sample_parameters(**self._init_args)
        )

    @abstractmethod
    def _sample_parameters(self, **kwargs):
        """
        Abstract method for sampling estimator layer parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward_pass(self, **kwargs):
        """
        Abstract method for defining layer forward pass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def backward_pass(self, **kwargs):
        """
        Abstract method for defining layer backward pass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class QuantumLayer(EstimatorLayer):
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            layers_num: Union[int, None] = 1,
            measurement_op: Callable = qml.PauliZ,
            measurements_num: int = 1,
            data_reuploading: bool = False
    ):
        super().__init__(
            init_args={
                'n_ansatz_params': ansatz.params_num,
                'layers_n': layers_num
            }
        )

        # Validate measurement operation
        if not self._is_valid_measurement_op(measurement_op):
            raise ValueError(
                "Invalid measurement operation provided."
            )

        # Validate if data reuploading is possible
        if data_reuploading and isinstance(feature_map, AmplitudeEmbedding):
            raise ValueError(
                "Data reuploading is not compatible with Amplitude Embedding ansatz. PennyLane "
                "does not allow to use multiple state preparation operations at the moment."
            )


        self._n_qubits = n_qubits
        self._ansatz = ansatz
        self._feature_map = feature_map
        self._layers_num = layers_num
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num
        self._data_reuploading = data_reuploading

        self._device = qml.device(
            name="default.qubit.jax", wires=self._n_qubits)

    def _sample_parameters(
            self,
            n_ansatz_params: Union[int, List[int]],
            layers_n: int = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Samples randomly quantum layer parameters.

        Args:
            n_ansatz_params: The number of parameters of the ansatz.
            layers_n: Number of layers in the quantum circuit.

        Returns:
            Dictionary with sampled parameters.
        """
        key = jax.random.PRNGKey(self._random_seed)

        if isinstance(n_ansatz_params, int):
            shape = (layers_n, n_ansatz_params) if layers_n else [n_ansatz_params]
        else:
            shape = (layers_n, *n_ansatz_params) if layers_n else [*n_ansatz_params]

        weights = 0.1 * jax.random.normal(key, shape=shape)

        return {'q_params': weights, 'c_params': None, 'batch_stats': None}

    @staticmethod
    def _is_valid_measurement_op(measurement_op):
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

    def quantum_circuit(
            self,
            x_data: Union[jnp.ndarray, None] = None,
            q_weights: Union[jnp.ndarray, None] = None
    ) -> None:
        """
        Applies the quantum circuit. This method is conditional on the data reuploading flag.
        If data reuploading is enabled, the feature map is applied at every layer.

        Args:
            x_data: Input data to be encoded into the quantum state.
            q_weights: Parameters for the variational form.
        """
        if not self._data_reuploading:
            self._feature_map.apply(features=x_data)
            for i in range(self._layers_num):
                self._ansatz.apply(params=q_weights[i])
        else:
            for i in range(self._layers_num):
                self._feature_map.apply(features=x_data)
                self._ansatz.apply(params=q_weights[i])

    def forward_pass(
            self,
            x_data: jnp.ndarray,
            q_params: jnp.ndarray,
            return_probs: Union[bool] = False
    ):
        """
        Forward pass method of the quantum layer returning quantum node outputs.

        Args:
            x_data: Input data array.
            q_params: Parameters for the quantum circuit.
            return_probs: Indicates whether the quantum model shall return probabilities.

        Returns:
            Outputs of the quantum node.
        """
        @jax.jit
        @qml.qnode(device=self._device, interface="jax")
        def _q_node():
            self.quantum_circuit(
                x_data=x_data, q_weights=q_params)

            if not return_probs:
                return [
                    qml.expval(self._measurement_op(i))
                    for i in range(self._measurements_num)
                ]
            else:
                return qml.probs(wires=range(self._n_qubits))
        return jnp.array(_q_node()).T

    def backward_pass(
            self,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
            loss_fn: Callable
    ) -> Tuple[float, jnp.ndarray]:
        """
        Backward pass method of the quantum layer returning loss value and gradients.

        Args:
            x_data: Input data array.
            y_data: Input data labels.
            loss_fn: Loss function used to calculate loss.

        Returns:
            Tuple of loss value and gradients.
        """
        def _calculate_loss(q_params, x, y):
            predictions = self.forward_pass(q_params=q_params, x_data=x)
            loss_val = loss_fn(predictions, y).mean()
            return loss_val

        loss, grads = jax.value_and_grad(
            _calculate_loss, argnums=0)(self.params.q_params, x_data, y_data)

        return loss, grads


class ClassicalLayer(EstimatorLayer):
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_module: nn.Module,
            batch_norm: bool = False
    ):
        super().__init__(
            init_args={
                'c_module': c_module,
                'input_shape': input_shape,
                'batch_norm': batch_norm
            }
        )
        self._c_module = c_module
        self._batch_norm = batch_norm

    def _sample_parameters(
            self,
            c_module: nn.Module,
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ):
        """
        Samples randomly classical layer parameters.

        Args:
            c_module: The classical model component.
            input_shape: The shape of the input data for the classical component of the hybrid model.
            batch_norm: Boolean indicating whether classical model uses batch normalization.

        Returns:
            Dictionary with sampled parameters.
        """
        inp_rng, init_rng = jax.random.split(
            jax.random.PRNGKey(seed=self._random_seed), num=2)

        if isinstance(input_shape, int):
            self.input_shape = (1, input_shape)
        else:
            self.input_shape = (1, *input_shape)

        c_inp = jax.random.normal(inp_rng, shape=self.input_shape)

        if batch_norm:
            variables = c_module.init(init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']
        else:
            variables = c_module.init(init_rng, c_inp)
            weights, batch_stats = variables['params'], None

        return {'q_params': None, 'c_params': weights, 'batch_stats': batch_stats}

    def forward_pass(
            self,
            x_data: jnp.ndarray,
            c_params: Dict[str, Mapping[str, jnp.ndarray]],
            batch_stats: Dict[str, Mapping[str, jnp.ndarray]],
            training: bool = False,
            flatten_output: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass method of the classical layer returning classical model outputs.

        Args:
            x_data: Input data.
            c_params: Parameters of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            training: Indicates whether the model is being used for training or inference.
            flatten_output: Indicates whether to flatten the output.

        Returns:
            Outputs of the classical model.
        """
        if self._batch_norm:
            if training:
                c_out, updates = self._c_module.apply(
                    {'params': c_params, 'batch_stats': batch_stats},
                    x_data, train=training, mutable=['batch_stats'])
                output = jax.numpy.array(c_out), updates['batch_stats']
            else:
                c_out = self._c_module.apply(
                    {'params': c_params,'batch_stats': batch_stats},
                    x_data, train=training, mutable=False)
                output = jax.numpy.array(c_out), None
        else:
            c_out = self._c_module.apply(
                {'params': c_params}, x_data)
            output = jax.numpy.array(c_out), None

        return output[0] if flatten_output else output

    def backward_pass(
            self,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
            loss_fn: Callable
    ) -> Tuple[float, Union[jnp.ndarray, Dict[str, Any]]]:
        """
        Backward pass method of the classical layer returning loss value and gradients.

        Args:
            x_data: Input data array.
            y_data: Input data labels.
            loss_fn: Loss function used to calculate loss.

        Returns:
            Tuple of loss value and gradients.
        """
        def _calculate_loss(c_params, batch_stats, x, y):
            predictions, self.params.batch_stats = self.forward_pass(
                x_data=x, c_params=c_params, batch_stats=batch_stats,
                training=True, flatten_output=False
            )
            loss_val = loss_fn(predictions, y).mean()
            return loss_val, self.params.batch_stats

        (loss, self.params.batch_stats), grads = jax.value_and_grad(
            _calculate_loss, argnums=0, has_aux=True
        )(self.params.c_params, self.params.batch_stats, x_data, y_data)

        return loss, grads


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
            layers: List[EstimatorLayer],
            estimator_type: str
    ):
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.estimator_type = estimator_type
        self.input_shape = None

        # Define estimator layers
        self.layers = layers

        # Initiate estimator optimizer
        self._trainer = self._init_trainer()

    def _init_trainer(self):
        """
        Initializes and returns an optimizer based on the specified estimator type.
        """
        if self.estimator_type == 'quantum':
            return QuantumOptimizer
        elif self.estimator_type == 'classical':
            return ClassicalOptimizer
        elif self.estimator_type == 'hybrid':
            return HybridOptimizer
        else:
            raise ValueError(
                f"Invalid optimizer type: {self.estimator_type},"
                f" available options are {'quantum', 'classical', 'hybrid'}"
            )

    def forward(
            self,
            x_data: jnp.ndarray,
            return_q_probs: bool = False,
            flatten_c_output: bool = False
    ):
        """
        Forward pass method of the entire estimator model.

        Args:
            x_data: Input data.
            return_q_probs: Indicates whether the quantum model shall return probabilities.
            flatten_c_output: Indicates whether to flatten the classical output.

        Returns:
            Output logits of the estimator.
        """
        output = x_data
        for layer in self.layers:
            q_params, c_params, batch_stats = layer.params

            if isinstance(layer, QuantumLayer):
                output = layer.forward_pass(
                    x_data=output, q_params=q_params, return_probs=return_q_probs)
            elif isinstance(layer, ClassicalLayer):
                output, batch_stats = layer.forward_pass(
                    x_data=output, c_params=c_params, batch_stats=batch_stats,
                    flatten_output=flatten_c_output)
                layer.params.batch_stats = batch_stats
            else:
                raise ValueError(
                    f"Estimator layer type not recognized: {type(layer)}"
                )

        return output


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
            model=self.forward,
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
