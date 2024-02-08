# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from typing import Union, Tuple

import flax.linen as nn
from jax import numpy as jnp

from fast_qml.machine_learning.estimator import HybridEstimator
from fast_qml.machine_learning.estimators.qnn import QNNRegressor, QNNClassifier
from fast_qml.machine_learning.estimators.vqa import VQRegressor, VQClassifier


class HybridRegressor(HybridEstimator):
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
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            q_model: Union[VQRegressor, QNNRegressor]
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model
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
        return jnp.array(self._model(
            c_weights=self._weights['c_weights'],
            q_weights=self._weights['q_weights'],
            x_data=x)
        ).ravel()


class HybridClassifier(HybridEstimator):
    """
    A hybrid quantum-classical classifier for classification tasks.

    This class extends the HybridEstimator for classification tasks using quantum neural networks
    or variational quantum algorithms combined with classical neural networks.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum classification model.

    Attributes:
        _classes_num: Number of classes in the classification task.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            q_model: Union[VQClassifier, QNNClassifier]
    ):
        super().__init__(
            input_shape=input_shape,
            c_model=c_model,
            q_model=q_model
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
        outputs = jnp.array(self._model(
            c_weights=self._weights['c_weights'],
            q_weights=self._weights['q_weights'],
            x_data=x)
        )
        if self._classes_num == 2:
            return outputs.ravel()
        else:
            return outputs.T

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

        if self._q_model.classes_num == 2:
            return jnp.where(predictions >= threshold, 1, 0)
        else:
            return jnp.argmax(predictions, axis=1)
