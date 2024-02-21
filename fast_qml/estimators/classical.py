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

from fast_qml.core.estimator import ClassicalEstimator

class ClassicalModel(ClassicalEstimator):
    """
    Specialized class for constructing and training classical neural network models.

    Inherits from ClassicalEstimator and focuses on leveraging classical computing resources for machine learning
    tasks. It provides a structured approach to model definition, parameter initialization, and training, facilitating
    the use of advanced features such as batch normalization and customizable optimization strategies.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
    """
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
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ) -> Union[jnp.ndarray, Dict[str, Any]]:
        """
        Initializes parameters for the classical and quantum models.

        Args:
            input_shape: The shape of the input data.
            batch_norm: Indicates whether batch normalization is used within the model.

        Returns:
            A dictionary containing initialized parameters.
        """
        if not all(isinstance(dim, int) for dim in input_shape):
            raise ValueError("input_shape must be a tuple or list of integers.")

        c_inp = jax.random.normal(self._inp_rng, shape=(1, *input_shape))

        if batch_norm:
            variables = self._c_model.init(self._init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']
            return {
                'weights': weights,
                'batch_stats': batch_stats
            }
        else:
            variables = self._c_model.init(self._init_rng, c_inp)
            weights = variables['params']
            return {
                'weights': weights
            }

    def _model(
            self,
            weights: Dict[str, Mapping[str, jnp.ndarray]],
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None],
            x_data: jnp.ndarray,
            training: bool
    ):
        """
        Defines the classical model inference.

        Args:
            weights: Weights of the classical model.
            batch_stats: Batch statistics for batch normalization.
            x_data: Input data for the model.
            training: Boolean flag indicating if training model inference.

        Returns:
            The output of the classical model.
        """
        def _classical_model():
            if self.batch_norm:
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
    """
    A classical regressor for regression tasks. This class extends the ClassicalModel for regression
    tasks using classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
    """
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
        if self.batch_norm:
            weights, batch_stats = (
                self.params['weights'], self.params['batch_stats'])
        else:
            weights, batch_stats = self.params['weights'], None

        return jnp.array(
            self._model(
                weights=weights, x_data=x,
                batch_stats=batch_stats, training=False)
        ).ravel()


class ClassicalClassifier(ClassicalModel):
    """
    A classical classifier for classification tasks. This class extends the ClassicalModel for
    classification tasks using classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
        classes_num: Number of classes in the classification problem.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer: Callable,
            batch_norm: bool,
            classes_num: int
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_norm=batch_norm
        )

        self.classes_num = classes_num

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
        if self.batch_norm:
            weights, batch_stats = (
                self.params['weights'], self.params['batch_stats'])
        else:
            weights, batch_stats = self.params['weights'], None

        logits = self._model(
            weights=weights, x_data=x,
            batch_stats=batch_stats, training=False
        )

        if self.classes_num == 2:
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

        if self.classes_num == 2:
            return jnp.where(logits >= threshold, 1, 0)
        else:
            return jnp.argmax(logits, axis=1)
