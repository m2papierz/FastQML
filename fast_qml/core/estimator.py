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
from functools import partial
from collections import OrderedDict
from pathlib import Path

from typing import Callable
from typing import Union
from typing import Dict
from typing import Mapping
from typing import Any
from typing import List
from typing import Tuple

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

import jax
import flax.linen as nn
import pennylane as qml
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class

from fast_qml.quantum_circuits.data_encoding import FeatureMap
from fast_qml.quantum_circuits.data_encoding import AmplitudeEmbedding
from fast_qml.quantum_circuits.ansatz import VariationalForm
from fast_qml.core.optimizer import ParametersOptimizer
from fast_qml.core.callbacks import EarlyStopping


@register_pytree_node_class
class EstimatorComponentParameters:
    """
    A class to hold parameters for an estimator component.

    Attributes:
        q_params: Quantum parameters.
        c_params: Classical parameters.
        batch_stats: Batch statistics for classical model.
    """
    def __init__(
            self,
            q_params: Union[ArrayLike, None],
            c_params: Union[ArrayLike, Dict[str, Any], None],
            batch_stats: Union[ArrayLike, Dict[str, Any], None]
    ):
        self.q_params = q_params
        self.c_params = c_params
        self.batch_stats = batch_stats

    @property
    def params_num(self):
        """
        Initializes the total_params attribute based on the weights and batch stats.
        """
        total_params = 0
        if self.q_params is not None:
            total_params += len(self.q_params.ravel())
        if self.c_params is not None:
            total_params += sum(x.size for x in jax.tree_leaves(self.c_params))
        return total_params


    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        children = []
        aux_data = {
            'q_params': self.q_params,
            'c_params': self.c_params,
            'batch_stats': self.batch_stats
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def __iter__(self):
        """
        Allow unpacking of the class instance.
        """
        yield self.q_params
        yield self.c_params
        yield self.batch_stats

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    q_params={self.q_params!r},\n"
            f"    c_params={self.c_params!r},\n"
            f"    batch_stats={self.batch_stats!r},\n"
            f"    total_params={self.params_num!r}\n)"
        )


class EstimatorComponent:
    """
    A base class for implementing custom estimator component.

    Attributes:
        _random_seed: A seed value for pseudo-random number generation.
        _init_args: Initialization arguments for the EstimatorComponentParameters instance.
        _parameters: An instance of EstimatorComponentParameters.
    """
    def __init__(
            self,
            init_args: Dict[str, Any]
    ):
        self._random_seed = 0
        self._init_args = init_args

        # Initiate estimator component parameters
        self._parameters = None
        self.init_parameters()

    @property
    def model_type(self) -> str:
        """
        Property returning the type of the estimator component.
        """
        return self.__class__.__name__

    @property
    def parameters(self) -> EstimatorComponentParameters:
        """
        Property returning estimator component parameters.
        """
        return self._parameters

    @property
    def params_num(self) -> int:
        """
        Returns the number total of parameters of the estimator component.
        """
        return self._parameters.params_num

    def init_parameters(self):
        """
        Initiates EstimatorComponentParameters instance for the estimator component.
        """
        # To ensure varied outcomes in pseudo-random sampling with JAX's explicit PRNG
        # system, we need to manually increment the random seed for each sampling
        self._random_seed += 1

        # Initiate component parameters with sampled parameters
        self._parameters = EstimatorComponentParameters(
            **self._sample_parameters(**self._init_args)
        )

    @abstractmethod
    def _sample_parameters(self, **kwargs):
        """
        Abstract method for sampling estimator component parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward_pass(self, **kwargs):
        """
        Abstract method for defining component forward pass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def backward_pass(self, **kwargs):
        """
        Abstract method for defining component backward pass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


@register_pytree_node_class
class QuantumModel(EstimatorComponent):
    """
    Implements a quantum component model that can be used in Estimator model.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        feature_map: The feature map to apply to the input data.
        ansatz: The variational form (ansatz) used in the circuit.
        layers_num: Number of times the ansatz is applied. Defaults to 1.
        measurement_op: Operation used for measurement. Defaults to `qml.PauliZ`.
        measurements_num: Number of measurements to perform. Defaults to 1.
        data_reuploading: Indicates if data reuploading is to be used. Defaults to False.
    """
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

        # TODO: Possibility of creating QuantumModel without feature map!

        self._n_qubits = n_qubits
        self._ansatz = ansatz
        self._feature_map = feature_map
        self._layers_num = layers_num
        self._measurement_op = measurement_op
        self._measurements_num = measurements_num
        self._data_reuploading = data_reuploading

    @property
    def measurements_num(self) -> int:
        """
        Property returning number of measurements in quantum circuit.
        """
        return self._measurements_num

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        aux_data = {
            'n_qubits': self._n_qubits,
            'feature_map': self._feature_map,
            'ansatz': self._ansatz,
            'layers_num': self._layers_num,
            'measurement_op': self._measurement_op,
            'measurements_num': self._measurements_num,
            'data_reuploading': self._data_reuploading
        }
        return [], aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def _sample_parameters(
            self,
            n_ansatz_params: Union[int, List[int]],
            layers_n: int = None
    ) -> Dict[str, Array]:
        """
        Samples randomly quantum model parameters.

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

        weights = 0.1 * jax.random.normal(
            key, shape=shape, dtype=jnp.float32)

        return {'q_params': weights, 'c_params': None, 'batch_stats': None}

    @staticmethod
    def _is_valid_measurement_op(measurement_op) -> bool:
        """
        Check if the provided measurement operation is valid.
        """
        return isinstance(measurement_op(0), qml.operation.Operation)

    def quantum_circuit(
            self,
            x_data: Union[ArrayLike, None] = None,
            q_weights: Union[ArrayLike, None] = None
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
                qml.Barrier(only_visual=True)
        else:
            for i in range(self._layers_num):
                self._feature_map.apply(features=x_data)
                self._ansatz.apply(params=q_weights[i])
                qml.Barrier(only_visual=True)

    def draw_circuit(
            self,
            device_expansion: bool = False
    ) -> None:
        """
        Draws the quantum circuit of the model.

        Args:
             device_expansion: Boolean flag indicating if to use 'device' expansion strategy.
        """
        if isinstance(self._feature_map, AmplitudeEmbedding):
            aux_input = jnp.zeros(2 ** self._n_qubits)
        else:
            aux_input = jnp.zeros(self._n_qubits)

        @qml.qnode(device=qml.device("default.qubit", self._n_qubits))
        def draw_q_node(data, params):
            self.quantum_circuit(data, params)
            return [
                qml.expval(self._measurement_op(i))
                for i in range(self._measurements_num)
            ]

        # Print the circuit drawing
        print(qml.draw(
            qnode=draw_q_node,
            expansion_strategy='device' if device_expansion else 'gradient',
            show_matrices=False)(aux_input, self.parameters.q_params)
        )

    @partial(jax.jit, static_argnums=(3,))
    def forward_pass(
            self,
            x_data: ArrayLike,
            q_params: ArrayLike,
            return_probs: Union[bool] = False
    ) -> Array:
        """
        Forward pass method of the quantum model returning quantum node outputs.

        Args:
            x_data: Input data array.
            q_params: Parameters for the quantum circuit.
            return_probs: Indicates whether the quantum model shall return probabilities.

        Returns:
            Outputs of the quantum node.
        """
        dev = qml.device(name="default.qubit.jax", wires=self._n_qubits)

        @qml.qnode(device=dev, interface="jax")
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

    @partial(jax.jit, static_argnums=(4,))
    def _compute_loss(
            self,
            q_params: ArrayLike,
            x_data: ArrayLike,
            y_data: ArrayLike,
            loss_fn: Callable
    ) -> Array:
        """
        Computes the loss of the model for a given batch of data. This method performs a forward
        pass using the model's parameters and the input data, and computes the loss using the provided
        loss function.

        Args:
            q_params: Parameters of the quantum model.
            x_data: Input data array.
            y_data: Target data array.
            loss_fn: The loss function to compute the loss between the predictions and the target data.

        Returns:
            The mean loss computed for the input data batch.
        """
        predictions =self.forward_pass(
            q_params=q_params, x_data=x_data)

        return loss_fn(predictions, y_data).mean()

    @partial(jax.jit, static_argnums=(3,))
    def backward_pass(
            self,
            x_data: ArrayLike,
            y_data: ArrayLike,
            loss_fn: Callable
    ) -> Tuple[float, Array]:
        """
        Backward pass method of the quantum model returning loss value and gradients computed
        in regard to quantum model parameters.

        Args:
            x_data: Input data array.
            y_data: Input data labels.
            loss_fn: Loss function used to calculate loss.

        Returns:
            Tuple of the loss value and the gradients.
        """
        # Compute gradients and the loss value for the batch of data
        loss, grads = jax.value_and_grad(
            fun=self._compute_loss,
            argnums=0)(self.parameters.q_params, x_data, y_data, loss_fn)

        return loss, grads


@register_pytree_node_class
class ClassicalModel(EstimatorComponent):
    """
    Implements a classical model for an estimator model.

    Args:
        input_shape: The shape of the input to the model.
        c_module: The Flax module.
        batch_norm: Indicates whether batch normalization is included in the model.
    """
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
        self._input_shape = input_shape
        self._c_module = c_module
        self._batch_norm = batch_norm

    @property
    def input_shape(self) -> Union[int, Tuple[int]]:
        """
        Property returning the shape of the input to the model.
        """
        return self._input_shape

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        aux_data = {
            'input_shape': self._input_shape,
            'c_module': self._c_module,
            'batch_norm': self._batch_norm,
        }
        return [], aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def _sample_parameters(
            self,
            c_module: nn.Module,
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ):
        """
        Samples randomly classical model parameters.

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
            self._input_shape = (1, input_shape)
        else:
            self._input_shape = (1, *input_shape)

        c_inp = jax.random.normal(
            inp_rng, shape=self._input_shape, dtype=jnp.float32)

        if batch_norm:
            variables = c_module.init(init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']
        else:
            variables = c_module.init(init_rng, c_inp)
            weights, batch_stats = variables['params'], None

        return {'q_params': None, 'c_params': weights, 'batch_stats': batch_stats}

    @partial(jax.jit, static_argnums=(4,))
    def forward_pass(
            self,
            x_data: ArrayLike,
            c_params: Dict[str, Mapping[str, ArrayLike]],
            batch_stats: Dict[str, Mapping[str, ArrayLike]],
            training: bool = False
    ) -> Tuple[Array, Union[Array, None]]:
        """
        Forward pass method of the classical model returning classical model outputs.

        Args:
            x_data: Input data.
            c_params: Parameters of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            training: Indicates whether the model is being used for training or inference.

        Returns:
            Outputs of the classical model.
        """
        bs_update = None

        if self._batch_norm:
            if training:
                c_out, updates = self._c_module.apply(
                    {'params': c_params, 'batch_stats': batch_stats},
                    x_data, train=training, mutable=['batch_stats'])
                bs_update = updates['batch_stats']
            else:
                c_out = self._c_module.apply(
                    {'params': c_params,'batch_stats': batch_stats},
                    x_data, train=training, mutable=False)
        else:
            c_out = self._c_module.apply({'params': c_params}, x_data)

        return jax.numpy.array(c_out), bs_update

    @partial(jax.jit, static_argnums=(5,))
    def _compute_loss(
            self,
            c_params: Dict[str, Mapping[str, ArrayLike]],
            batch_stats: Dict[str, Mapping[str, ArrayLike]],
            x_data: ArrayLike,
            y_data: ArrayLike,
            loss_fn: Callable
    ) -> Array:
        """
        Computes the loss of the model for a given batch of data.

        This method performs a forward pass using the model's parameters and the input data,
        computes the loss using the provided loss function, and updates the batch statistics
        if applicable.

        Args:
            c_params: Parameters of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            x_data: Input data array.
            y_data: Target data array.
            loss_fn: The loss function to compute the loss between the predictions and the target data.

        Returns:
            The mean loss computed for the input data batch.
        """
        predictions, batch_stats = self.forward_pass(
            x_data=x_data,
            c_params=c_params,
            batch_stats=batch_stats,
            training=True
        )

        # Update batch statistics if applicable
        if batch_stats:
            self.parameters.batch_stats = batch_stats

        return loss_fn(predictions, y_data).mean()

    @partial(jax.jit, static_argnums=(3,))
    def backward_pass(
            self,
            x_data: ArrayLike,
            y_data: ArrayLike,
            loss_fn: Callable
    ) -> Tuple[float, Union[Array, Dict[str, Any]]]:
        """
        Backward pass method of the classical model returning loss value and gradients in
        regard to the parameters of the model.

        Args:
            x_data: Input data array.
            y_data: Input data labels.
            loss_fn: Loss function used to calculate loss.

        Returns:
            Tuple of the loss value and the gradients.
        """
        # Unpack model parameters
        _, c_params, batch_stats = self.parameters

       # Compute gradients and the loss value for the batch of data
        loss, grads = jax.value_and_grad(
            self._compute_loss, argnums=0
        )(c_params, batch_stats, x_data, y_data, loss_fn)

        return loss, grads


@register_pytree_node_class
class Estimator:
    def __init__(
            self,
            estimator_components: Union[EstimatorComponent, List[EstimatorComponent]],
            loss_fn: Callable,
            optimizer_fn: Callable
    ):
        self._loss_fn = loss_fn
        self._optimizer_fn = optimizer_fn

        # If single model is provided put it into list for compatibility
        # with class methods
        if isinstance(estimator_components, list):
            self.estimator_components = estimator_components
        else:
            self.estimator_components = [estimator_components]

        # Validate provided component models types
        for model in self.estimator_components:
            if not isinstance(model, EstimatorComponent):
                raise TypeError(
                    f"Invalid estimator component type: {type(model)}"
                )

        # Initiate estimator parameters
        self._parameters = OrderedDict()
        self.init_parameters(resample=False)

    @property
    def parameters(self) -> OrderedDict:
        """
        Property returning estimator parameters.
        """
        return self._parameters

    @property
    def q_parameters(self) -> OrderedDict:
        """
        Property returning estimator quantum parameters.
        """
        q_parameters = OrderedDict()
        for key, value in self.parameters.items():
            if key.startswith(QuantumModel.__name__):
                q_parameters[key] = value
        return q_parameters

    @property
    def c_parameters(self) -> OrderedDict:
        """
        Property returning estimator classical parameters.
        """
        c_parameters = OrderedDict()
        for key, value in self.parameters.items():
            if key.startswith(ClassicalModel.__name__):
                c_parameters[key] = value
        return c_parameters

    @property
    def params_num(self) -> int:
        """
        Returns the number total of parameters of the Estimator.
        """
        return sum(model.params_num for model in self.estimator_components)

    @property
    def outputs_num(self) -> int:
        """
        Returns the number of outputs of the Estimator.
        """
        last_model = self.estimator_components[-1]
        if isinstance(last_model, QuantumModel):
            outputs_num = last_model.measurements_num
            return outputs_num
        else:
            last_layer = list(last_model.parameters.c_params.keys())[-1]
            outputs_num = last_model.parameters.c_params[
                last_layer]['kernel'].shape[-1]
            return outputs_num


    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        aux_data = {
            'estimator_components': self.estimator_components,
            'loss_fn': self._loss_fn,
            'optimizer_fn': self._optimizer_fn
        }
        return [], aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def init_parameters(
            self,
            resample: bool = False
    ) -> None:
        """
        Initiates Estimator parameters as OrderedDict holding parameters of
        each Estimator component.

        Args:
            resample: Indicates whether to resample the estimator components parameters.

        Returns:
            Tuple of OrderedDict holding parameters of the Estimator.
        """
        q_idx, c_idx = 0, 0

        for m in self.estimator_components:
            if resample:
                m.init_parameters()
            q_params, c_params, batch_stats = m.parameters

            if isinstance(m, QuantumModel):
                self._parameters[f"{m.model_type}{q_idx}"] = q_params
                q_idx += 1

            if isinstance(m, ClassicalModel):
                if batch_stats is not None:
                    self._parameters[f"{m.model_type}{c_idx}"] = [c_params, batch_stats]
                else:
                    self._parameters[f"{m.model_type}{c_idx}"] = [c_params, None]
                c_idx += 1

    def _update_parameters(
            self,
            q_parameters: Union[OrderedDict, None] = None,
            c_parameters: Union[OrderedDict, None] = None
    ) -> None:
        """
        Updates the estimator parameters with the provided quantum and classical parameters.

        Args:
            q_parameters: OrderedDict with updated quantum parameters.
            c_parameters: OrderedDict with updated classical parameters.
        """
        for key in self.parameters.keys():
            if key in q_parameters:
                self.parameters[key] = q_parameters[key]
            elif key in c_parameters:
                self.parameters[key] = c_parameters[key]

    @partial(jax.jit, static_argnums=(4,))
    def forward_pass(
            self,
            x_data: Array,
            q_parameters: OrderedDict,
            c_parameters: OrderedDict,
            return_q_probs: bool = False
    ):
        """
        Forward pass method of the entire estimator model.

        Args:
            x_data: Input data.
            q_parameters: Parameters of the quantum models.
            c_parameters: Parameters of the classical models.
            return_q_probs: Indicates whether the quantum model shall return probabilities.

        Returns:
            Output logits of the estimator.
        """
        output = x_data
        q_count, c_count = 0, 0
        for idx, model in enumerate(self.estimator_components):
            if isinstance(model, QuantumModel):
                # Unpack the quantum parameters
                q_params = q_parameters[f"{model.model_type}{q_count}"]

                # Probabilities can only be returned in the last layer
                if (idx + 1) == len(self.estimator_components):
                    return_q_probs = return_q_probs
                else:
                    return_q_probs = False

                # Forward pas of the quantum model
                output = model.forward_pass(
                    x_data=output, q_params=q_params, return_probs=return_q_probs)
                q_count += 1

                # Transpose output of the quantum model when probabilities are returned
                if return_q_probs:
                    output = jnp.transpose(output)

            elif isinstance(model, ClassicalModel):
                # Unpack the classical parameters
                c_params, batch_stats = c_parameters[f"{model.model_type}{c_count}"]

                # Forward pas of the classical model
                output, _ = model.forward_pass(
                    x_data=output, c_params=c_params, batch_stats=batch_stats)
                c_count += 1
            else:
                raise ValueError(
                    f"Estimator component type not recognized: {type(model)}"
                )

        return output

    @jax.jit
    def _compute_loss(
            self,
            q_parameters: OrderedDict,
            c_parameters: OrderedDict,
            x_data: ArrayLike,
            y_data: ArrayLike
    ) -> Array:
        """
        Computes the loss of the estimator for a given batch of data.

        Args:
            q_parameters: Parameters of the quantum models.
            c_parameters: Parameters of the classical models.
            x_data: Input data array.
            y_data: Target data array.

        Returns:
            The mean loss computed for the input data batch.
        """
        predictions = self.forward_pass(
            x_data=x_data,
            q_parameters=q_parameters,
            c_parameters=c_parameters,
            return_q_probs=False
        )

        return self._loss_fn(predictions, y_data).mean()

    @jax.jit
    def backward_pass(
            self,
            x_data: ArrayLike,
            y_data: ArrayLike
    ) -> Tuple[float, OrderedDict, OrderedDict]:
        """
        Backward pass method of the estimator returning loss value and gradients.

        Args:
            x_data: Input data array.
            y_data: Input data labels.

        Returns:
            Tuple of the loss value and the gradients.
        """
        # Compute gradients and the loss value for the batch of data
        loss, (q_grads, c_grads) = jax.value_and_grad(
            self._compute_loss, argnums=(0, 1)
        )(self.q_parameters, self.c_parameters, x_data=x_data, y_data=y_data)

        return loss, q_grads, c_grads

    def fit(
            self,
            train_data: Union[np.ndarray, torch.Tensor, DataLoader],
            val_data: Union[np.ndarray, torch.Tensor, DataLoader],
            train_targets: Union[np.ndarray, torch.Tensor, None] = None,
            val_targets: Union[np.ndarray, torch.Tensor, None] = None,
            c_learning_rate: float = 0.0001,
            q_learning_rate: float = 0.01,
            num_epochs: int = 500,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            verbose: bool = True
    ) -> None:
        """
        Fits estimator parameters to training data with objective of estimator optimization

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.

        Args:
            train_data: Input features for training.
            train_targets: Target outputs for training.
            val_data: Input features for validation.
            val_targets: Target outputs for validation.
            c_learning_rate: Learning rate for the classical optimizer.
            q_learning_rate: Learning rate for the quantum optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            early_stopping: Instance of EarlyStopping to be used during training.
            verbose : If True, prints verbose messages during training.
        """
        optimizer = ParametersOptimizer(
            q_parameters=self.q_parameters,
            c_parameters=self.c_parameters,
            forward_fn=self.forward_pass,
            loss_fn=self._loss_fn,
            q_optimizer=self._optimizer_fn(q_learning_rate),
            c_optimizer=self._optimizer_fn(c_learning_rate),
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

        self._update_parameters(*optimizer.parameters)

    def predict_proba(
            self,
            x: ArrayLike
    ) -> Array:
        """
        Predict the probability of each class for the given input data. The output probabilities
        indicate the likelihood of each class for each sample.

        Args:
            x: An array of input data.

        Returns:
            An array of predicted probabilities. For binary classification, this will be a 1D array with
            a single probability for each sample. For multi-class classification, this will be a 2D array
            where each row corresponds to a sample and each column corresponds to a class.
        """
        logits = self.forward_pass(
            x_data=x,
            q_parameters=self.q_parameters,
            c_parameters=self.c_parameters,
            return_q_probs=False
        )

        if self.outputs_num == 2:
            return jnp.array(logits).ravel()
        else:
            return jnp.array(logits)

    def predict(
            self,
            x: ArrayLike,
            threshold: float = 0.5,
    ) -> Array:
        """
        Predict class labels for the given input data.

        For binary classification, the function applies a threshold to the output probabilities to
        determine the class labels. For multi-class classification, the function assigns each sample
        to the class with the highest probability.

        Args:
            x: An array of input data
            threshold: The threshold for converting probabilities to binary class labels. Defaults to 0.5.

        Returns:
            An array of predicted class labels. For binary classification, this will be a 1D array with
            binary labels (0 or 1). For multi-class classification, this will be a 1D array where each
            element is the predicted class index.
        """
        x = jnp.array(x, dtype=jnp.float32)
        logits = self.predict_proba(x)

        if self.outputs_num == 2:
            return jnp.where(logits >= threshold, 1, 0)
        else:
            return jnp.argmax(logits, axis=1)

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
            pickle.dump(self._parameters, f)

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
            self._parameters = pickle.load(f)
