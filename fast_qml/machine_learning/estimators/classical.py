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
    Callable, Union, Any, Tuple, Dict, Mapping)

import jax
import flax.linen as nn
from jax import numpy as jnp

from fast_qml.machine_learning.estimator import ClassicalEstimator

class ClassicalModel(ClassicalEstimator):
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
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
            return {
                'weights': variables['params'],
                'batch_stats': variables['batch_stats']
            }
        else:
            variables = self._c_model.init(self._init_rng, c_inp)
            return {
                'weights': variables['params']
            }

    def _model(
            self,
            weights: Dict[str, Mapping[str, jnp.ndarray]],
            x_data: jnp.ndarray,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            training: bool
    ):
        """
        Defines the classical model inference.

        Args:
            weights: Weights of the classical model.
            x_data: Input data for the model.

        Returns:
            The output of the hybrid model.
        """
        def _classical_model():
            if self._batch_norm:
                if training:
                    c_out, updates = self._c_model.apply(
                        {'params': weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=['batch_stats'])
                    return jax.numpy.array(c_out), updates['batch_stats']
                else:
                    c_out = self._c_model.apply(
                        {'params': weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=False)
                    return jax.numpy.array(c_out)
            else:
                c_out = self._c_model.apply({'params': weights}, x_data)
                return jax.numpy.array(c_out)

        return _classical_model()


class ClassicalRegressor(ClassicalModel):
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
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
            weights, batch_stats = (
                self._params['weights'], self._params['batch_stats'])
        else:
            weights, batch_stats = self._params['weights'], None

        return jnp.array(
            self._model(
                weights=weights, x_data=x,
                batch_stats=batch_stats, training=False)
        ).ravel()


class ClassicalClassifier(ClassicalModel):
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool,
            num_classes: int
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_norm=batch_norm
        )

        self.num_classes = num_classes

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
            weights, batch_stats = (
                self._params['weights'], self._params['batch_stats'])
        else:
            weights, batch_stats = self._params['weights'], None

        logits = self._model(
            weights=weights, x_data=x,
            batch_stats=batch_stats, training=False
        )

        if self.num_classes == 2:
            return jnp.array(logits).ravel()
        else:
            return jnp.array(logits)

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
        logits = self.predict_proba(x)

        if self.num_classes == 2:
            return jnp.where(logits >= threshold, 1, 0)
        else:
            return jnp.argmax(logits, axis=1)