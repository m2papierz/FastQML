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
    Union, Dict, Any, Tuple, Callable, Mapping)

import jax
import flax.linen as nn
import pennylane as qml
from jax import numpy as jnp

from fast_qml.machine_learning.estimator import HybridEstimator
from fast_qml.machine_learning.estimators.qnn import QNNRegressor, QNNClassifier
from fast_qml.machine_learning.estimators.vqa import VQRegressor, VQClassifier


class HybridModel(HybridEstimator):
    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQRegressor, VQClassifier, QNNRegressor, QNNClassifier],
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_norm=batch_norm
        )

    def _initialize_parameters(
            self,
            input_shape: Union[int, Tuple[int]],
            batch_norm: bool = False
    ) -> Dict[str, Any]:
        """
        Initializes weights for the classical and quantum models.

        Args:
            input_shape: The shape of the input data.

        Returns:
            A dictionary containing initialized weights for both classical and quantum models.
        """
        if not all(isinstance(dim, int) for dim in input_shape):
            raise ValueError("input_shape must be a tuple or list of integers.")

        c_inp = jax.random.normal(self._inp_rng, shape=(1, *input_shape))

        if batch_norm:
            variables = self._c_model.init(self._init_rng, c_inp, train=False)
            c_weights, batch_stats = variables['params'], variables['batch_stats']
            return {
                'c_weights': c_weights,
                'q_weights': self._q_model.weights,
                'batch_stats': batch_stats
            }
        else:
            variables = self._c_model.init(self._init_rng, c_inp)
            c_weights = variables['params']
            return {
                'c_weights': c_weights,
                'q_weights': self._q_model.weights
            }

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

        Args:
            c_weights: Weights of the classical model.
            q_weights: Weights of the quantum model.
            x_data: Input data for the model.

        Returns:
            The output of the hybrid model.
        """
        def _hybrid_model():
            if self._batch_norm:
                if training:
                    c_out, updates = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=['batch_stats'])
                    q_out = self._q_model.q_model(
                        weights=q_weights, x_data=jax.numpy.array(c_out))
                    return jax.numpy.array(q_out), updates['batch_stats']
                else:
                    c_out = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=False)
                    q_out = self._q_model.q_model(
                        weights=q_weights, x_data=jax.numpy.array(c_out))
                    return jax.numpy.array(q_out)
            else:
                c_out = self._c_model.apply({'params': c_weights}, x_data)
                q_out = self._q_model.q_model(
                    weights=q_weights, x_data=jax.numpy.array(c_out))
                return jax.numpy.array(q_out)

        return _hybrid_model()


class HybridRegressor(HybridModel):
    """
    A hybrid quantum-classical regressor for regression tasks.

    This class extends the HybridEstimator for regression tasks using quantum neural networks
    or variational quantum algorithms combined with classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum regression model.
    """

    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQRegressor, QNNRegressor],
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_norm=batch_norm
        )

    def predict(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the predictions (model outputs) for the given input data.

        Args:
            x: An array of input data.
        """
        if self._batch_norm:
            c_weights, batch_stats = (
                self._params['c_weights'], self._params['batch_stats'])
        else:
            c_weights, batch_stats = self._params['c_weights'], None
        q_weights = self._params['q_weights']

        return jnp.array(
            self._model(
                c_weights=c_weights, q_weights=q_weights,
                x_data=x, batch_stats=batch_stats, training=False)
        ).ravel()


class HybridClassifier(HybridModel):
    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQClassifier, QNNClassifier],
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_norm=batch_norm
        )

        self._num_classes = self._q_model.num_classes

    def predict_proba(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
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
        if self._batch_norm:
            c_weights, batch_stats = (
                self._params['c_weights'], self._params['batch_stats'])
        else:
            c_weights, batch_stats = self._params['c_weights'], None
        q_weights = self._params['q_weights']

        logits = self._model(
            c_weights=c_weights, q_weights=q_weights,
            x_data=x, batch_stats=batch_stats, training=False
        )

        if self._num_classes == 2:
            return jnp.array(logits).ravel()
        else:
            return jnp.array(logits).T

    def predict(
            self,
            x: jnp.ndarray,
            threshold: float = 0.5
    ) -> jnp.ndarray:
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
        predictions = self.predict_proba(x)

        if self._num_classes == 2:
            return jnp.where(predictions >= threshold, 1, 0)
        else:
            return jnp.argmax(predictions, axis=1)
