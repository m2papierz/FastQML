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
    Union, Any, Tuple, Dict, Mapping
)

import jax
import flax.linen as nn
from jax import numpy as jnp

from fast_qml.machine_learning.optimizer import HybridOptimizer
from fast_qml.machine_learning.callbacks import EarlyStopping
from fast_qml.machine_learning.estimators.qnn import QNNRegressor, QNNClassifier
from fast_qml.machine_learning.estimators.vqa import VQRegressor, VQClassifier


class HybridEstimator:
    """
    Base class for creating hybrid quantum-classical machine learning models.

    This class combines classical neural network models with quantum neural networks or variational
    quantum algorithms to form a hybrid model for regression or classification tasks.

    Attributes:
        _c_model: The classical neural network model.
        _q_model: The quantum model, which can be either a quantum neural network or a variational quantum algorithm.
        _optimizer: Optimizer for training the hybrid model.
        _inp_rng: Random number generator key for input initialization.
        _init_rng: Random number generator key for model initialization.
        _weights: Initialized weights for both classical and quantum models.

    Args:
        input_shape: The shape of the input data.
        c_model: The classical neural network model.
        q_model: The quantum model.
    """
    def __init__(
            self,
            input_shape: Union[int, Tuple[int]],
            c_model: nn.Module,
            q_model: Union[VQRegressor, VQRegressor, QNNRegressor, QNNClassifier],
    ):
        self._c_model = c_model
        self._q_model = q_model

        self._optimizer = HybridOptimizer

        self._inp_rng, self._init_rng = jax.random.split(
            jax.random.PRNGKey(seed=42), num=2)
        self._weights = self._init_weights(input_shape)

    def _init_weights(self, input_shape) -> Dict[str, Any]:
        """
        Initializes weights for the classical and quantum models.

        Args:
            input_shape: The shape of the input data.

        Returns:
            A dictionary containing initialized weights for both classical and quantum models.
        """
        c_inp = jax.random.normal(self._inp_rng, shape=input_shape)
        c_weights = self._c_model.init(self._init_rng, c_inp)
        return {
            'c_weights': c_weights,
            'q_weights': self._q_model.weights
        }

    def _model(
            self,
            c_weights: Dict[str, Mapping[str, jnp.ndarray]],
            q_weights: jnp.ndarray,
            x_data: jnp.ndarray
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
            c_out = self._c_model.apply(c_weights, x_data)
            c_out = jax.numpy.array(c_out)
            q_out = self._q_model.q_model(weights=q_weights, x_data=c_out)
            return q_out
        return _hybrid_model()

    def fit(
            self,
            x_train: jnp.ndarray,
            y_train: jnp.ndarray,
            x_val: jnp.ndarray = None,
            y_val: jnp.ndarray = None,
            learning_rate: float = 0.01,
            num_epochs: int = 500,
            batch_size: int = None,
            early_stopping: EarlyStopping = None,
            verbose: bool = True
    ) -> None:
        """
        Trains the hybrid classical-quantum model on the given data.

        This method optimizes the weights of the hybrid model using the specified loss function
        and optimizer. It updates the weights based on the training data over a number of epochs.

        Args:
            x_train: Input features for training.
            y_train: Target outputs for training.
            x_val: Input features for validation.
            y_val: Target outputs for validation.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to run the training.
            batch_size: Size of batches for training. If None, the whole dataset is used in each iteration.
            early_stopping: Instance of EarlyStopping to be used during training.
            verbose : If True, prints verbose messages during training.

        If early stopping is configured and validation data is provided, the training process will
        stop early if no improvement is seen in the validation loss for a specified number of epochs.
        """
        optimizer = self._optimizer(
            c_params=self._weights['c_weights'],
            q_params=self._weights['q_weights'],
            model=self._model,
            loss_fn=self._q_model.loss_fn,
            batch_size=batch_size,
            epochs_num=num_epochs,
            learning_rate=learning_rate,
            early_stopping=early_stopping
        )

        optimizer.optimize(
            x_train=jnp.array(x_train),
            y_train=jnp.array(y_train),
            x_val=jnp.array(x_val),
            y_val=jnp.array(y_val),
            verbose=verbose
        )

        self._weights['c_weights'], self._weights['q_weights'] = optimizer.weights


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
