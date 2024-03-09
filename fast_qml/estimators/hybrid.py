# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from typing import Union
from typing import Dict
from typing import Mapping
from typing import Tuple

import jax
import flax.linen as nn
from jax import numpy as jnp

from fast_qml.core.estimator import Estimator
from fast_qml.estimators.quantum import QNNRegressor
from fast_qml.estimators.quantum import QNNClassifier
from fast_qml.estimators.quantum import VQRegressor
from fast_qml.estimators.quantum import VQClassifier


class HybridEstimator(Estimator):
    """
    Base class for hybrid quantum-classical machine learning models.

    Inherits from HybridEstimator and facilitates the integration of classical neural network architectures
    with quantum computing models to create powerful hybrid models. This approach aims to leverage the
    computational benefits of quantum processing for specific tasks within a broader machine learning workflow,
    potentially improving performance on tasks where quantum computing offers an advantage.

    The HybridModel class is designed to be flexible and extendable, serving as a foundation for building
    various types of hybrid models tailored to specific applications.

    Args:
        input_shape: The shape of the input data for the classical component of the hybrid model.
        c_model: The classical model component.
        q_model: The quantum model component, defined as an instance of a QuantumEstimator subclass.
        batch_norm: Boolean indicating whether classical model uses batch normalization.
    """
    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQRegressor, VQClassifier, QNNRegressor, QNNClassifier],
            batch_norm: bool = False
    ):
        self._c_model = c_model
        self._q_model = q_model
        self._batch_norm = batch_norm
        q_weights = q_model.params.q_weights

        super().__init__(
            loss_fn=q_model.loss_fn,
            optimizer_fn=q_model.optimizer_fn,
            estimator_type='hybrid',
            init_args={
                'c_model': c_model,
                'q_weights': q_weights,
                'input_shape': input_shape,
                'batch_norm': batch_norm
            }
        )

    def _sample_parameters(
            self,
            c_model: nn.Module,
            q_weights: Union[int, Tuple[int]],
            input_shape: Union[int, Tuple[int], None] = None,
            batch_norm: Union[bool, None] = None
    ):
        """
        Samples randomly estimator parameters.

        Args:
            c_model: The classical model component.
            q_weights: The weights of the quantum model.
            input_shape: The shape of the input data for the classical component of the hybrid model.
            batch_norm: Boolean indicating whether classical model uses batch normalization.

        Returns:
            Dictionary with sampled parameters.
        """
        inp_rng, init_rng = jax.random.split(
            jax.random.PRNGKey(seed=self.random_seed), num=2)

        if isinstance(input_shape, int):
            shape = (1, input_shape)
        else:
            shape = (1, *input_shape)

        c_inp = jax.random.normal(inp_rng, shape=shape)

        if batch_norm:
            variables = c_model.init(init_rng, c_inp, train=False)
            weights, batch_stats = variables['params'], variables['batch_stats']
            return {
                'c_weights': weights,
                'q_weights': q_weights,
                'batch_stats': batch_stats}
        else:
            variables = c_model.init(init_rng, c_inp)
            weights = variables['params']
            return {
                'c_weights': weights,
                'q_weights': q_weights
            }

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
        Defines estimator model.

        Args:
            x_data: Input data.
            q_weights: Weights of the quantum model.
            c_weights: Weights of the classical model.
            batch_stats: Batch normalization statistics for the classical model.
            training: Indicates whether the model is being used for training or inference.
            q_model_probs: Indicates whether the quantum model shall return probabilities.

        Returns:
            Outputs of the estimator model.
        """
        def _hybrid_model():
            if self._batch_norm:
                if training:
                    c_out, updates = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=['batch_stats'])
                    q_out = self._q_model.model(
                        q_weights=q_weights,
                        x_data=jax.numpy.array(c_out),
                        q_model_probs=q_model_probs)
                    return jax.numpy.array(q_out), updates['batch_stats']
                else:
                    c_out = self._c_model.apply(
                        {'params': c_weights, 'batch_stats': batch_stats},
                        x_data, train=training, mutable=False)
                    q_out = self._q_model.model(
                        q_weights=q_weights,
                        x_data=jax.numpy.array(c_out),
                        q_model_probs=q_model_probs)
                    return jax.numpy.array(q_out)
            else:
                c_out = self._c_model.apply({'params': c_weights}, x_data)
                q_out = self._q_model.model(
                    q_weights=q_weights,
                    x_data=jax.numpy.array(c_out),
                    q_model_probs=q_model_probs)
                return jax.numpy.array(q_out)

        return _hybrid_model()


class HybridRegressor(HybridEstimator):
    """
    A hybrid quantum-classical regressor for regression tasks.

    This class extends the HybridEstimator for regression tasks using quantum neural networks
    or variational quantum algorithms combined with classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum regression model.
        batch_norm: Boolean indicating whether classical model uses batch normalization.
    """
    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQRegressor, QNNRegressor],
            batch_norm: bool = False
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model,
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
                q_weights=self.params.q_weights,
                batch_stats=self.params.batch_stats,
                x_data=x, training=False)
        ).ravel()


class HybridClassifier(HybridEstimator):
    """
    A hybrid quantum-classical classifier for classification tasks.

    This class extends the HybridEstimator for classification tasks using quantum neural networks
    or variational quantum algorithms combined with classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum classifier model.
        batch_norm: Boolean indicating whether classical model uses batch normalization.
    """
    def __init__(
            self,
            input_shape,
            c_model: nn.Module,
            q_model: Union[VQClassifier, QNNClassifier],
            batch_norm: bool = False
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model,
            batch_norm=batch_norm
        )

        self._classes_num = self._q_model.classes_num

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
                q_weights=self.params.q_weights,
                batch_stats=self.params.batch_stats,
                x_data=x, training=False
        )

        if self._classes_num == 2:
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

        if self._classes_num == 2:
            return jnp.where(predictions >= threshold, 1, 0)
        else:
            return jnp.argmax(predictions, axis=1)
