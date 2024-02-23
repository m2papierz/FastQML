# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from typing import Callable, Union, Tuple, Dict, Mapping

import jax
import flax.linen as nn
from jax import numpy as jnp

from fast_qml.core.estimator import EstimatorParameters, Estimator

class ClassicalEstimator(Estimator):
    """
    Specialized class for constructing and training classical neural network models.

    Inherits from ClassicalEstimator and focuses on leveraging classical computing resources for machine learning
    tasks. It provides a structured approach to model definition, parameter initialization, and training, facilitating
    the use of advanced features such as batch normalization and customizable optimization strategies.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer_fn: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer_fn: Callable,
            batch_norm: bool
    ):
        super().__init__(
            loss_fn=loss_fn, optimizer_fn=optimizer_fn, estimator_type='classical'
        )

        self.params = EstimatorParameters(
            **self._init_parameters(
                c_model=c_model,
                input_shape=input_shape,
                batch_norm=batch_norm
            )
        )

        self._c_model = c_model
        self.batch_norm = batch_norm

    def _init_parameters(
            self,
            c_model: nn.Module,
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ) :
        if isinstance(input_shape, int):
            shape = (1, input_shape)
        else:
            shape = (1, *input_shape)

        c_inp = jax.random.normal(self._inp_rng, shape=shape)

        if batch_norm:
            variables = c_model.init(self._init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']
            return {'c_weights': weights, 'batch_stats': batch_stats}
        else:
            variables = c_model.init(self._init_rng, c_inp)
            weights = variables['params']
            return {'c_weights': weights}

    def model(
            self,
            x_data: jnp.ndarray,
            q_weights: Union[jnp.ndarray, None] = None,
            c_weights: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            batch_stats: Union[Dict[str, Mapping[str, jnp.ndarray]], None] = None,
            training: Union[bool, None] = None
    ):
        """
        Defines estimator model.

        Args:
            x_data: Input data.
            q_weights: Weights of the quantum model.
            c_weights: Weights of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            training: Specifies whether the model is being used for training or inference.

        Returns:
            Outputs of the estimator model.
        """
        def _classical_model():
            if self.batch_norm:
                if training:
                    c_out, updates = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=['batch_stats'])
                    return jax.numpy.array(c_out), updates['batch_stats']
                else:
                    c_out = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=False)
                    return jax.numpy.array(c_out)
            else:
                c_out = self._c_model.apply({'params': c_weights}, x_data)
                return jax.numpy.array(c_out)

        return _classical_model()


class ClassicalRegressor(ClassicalEstimator):
    """
    A classical regressor for regression tasks. This class extends the ClassicalModel for regression
    tasks using classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer_fn: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer_fn: Callable,
            batch_norm: bool
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
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
        return jnp.array(
            self.model(
                c_weights=self.params.c_weights,
                batch_stats=self.params.batch_stats,
                x_data=x, training=False)
        ).ravel()


class ClassicalClassifier(ClassicalEstimator):
    """
    A classical classifier for classification tasks. This class extends the ClassicalModel for
    classification tasks using classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        loss_fn: The loss function used to evaluate the model.
        optimizer_fn: The optimization algorithm.
        batch_norm: Indicates whether batch normalization is used within the model.
        classes_num: Number of classes in the classification problem.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            loss_fn: Callable,
            optimizer_fn: Callable,
            batch_norm: bool,
            classes_num: int
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
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
        logits = self.model(
            c_weights=self.params.c_weights,
            batch_stats=self.params.batch_stats,
            x_data=x, training=False
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
