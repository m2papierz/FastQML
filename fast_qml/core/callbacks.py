# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from jax import numpy as jnp


class EarlyStopping:
    """
    EarlyStopping utility to stop the training when a monitored quantity stops improving.

    Attributes:
        patience: Number of epochs with no improvement after which training will be stopped.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        best_loss: The best loss observed within the training process.
        wait: The number of epochs that have passed without improvement.
        stopped_epoch: Indicates the epoch at which the training was stopped.
        stop_training: Flag that indicates whether the training should be stopped.
    """

    def __init__(
            self,
            patience: int = 30,
            min_delta: float = 0.00001
    ):
        """
        Initializes the EarlyStopping instance with default or custom values.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def __call__(
            self,
            current_loss: float
    ) -> None:
        """
        Call method to update the state of the early stopping mechanism.

        This method should be called at the end of each epoch, passing in the current loss value.
        If the loss does not improve for a number of epochs equal to 'patience', sets the 'stop_training'
        flag to True.

        Args:
            current_loss: The loss value for the current epoch.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = True
                self.stop_training = True


class BestModelCheckpoint:
    """
    A callback class for tracking and storing the best model parameters based on validation loss during training.

    This class is designed to be used with an optimizer during the training process. It keeps track of the best
    model parameters when provided with the current model parameters and validation loss at each epoch.

    Attributes:
        best_q_params: Stores the best quantum model parameters encountered during training.
        best_c_params: Optionally stores the best classical model parameters if provided during training.
        best_val_loss: Records the lowest validation loss encountered during training.
    """
    def __init__(self):
        self.best_q_params = None
        self.best_c_params = None
        self.best_val_loss = float('inf')

    def update(
            self,
            current_val_loss: float,
            current_c_params: jnp.ndarray = None,
            current_q_params: jnp.ndarray = None,
    ) -> None:
        """
        Updates the best model parameters if the current validation loss is lower.

        Args:
            current_q_params: Current Q model parameters.
            current_val_loss: Current validation loss.
            current_c_params: Optional current C model parameters.
        """
        if current_val_loss < self.best_val_loss:
            if current_q_params is not None:
                self.best_q_params = current_q_params
            if current_c_params is not None:
                self.best_c_params = current_c_params
            self.best_val_loss = current_val_loss

    def load_best_model(self, optimizer) -> None:
        """
        Loads the best model parameters into the optimizer.

        Args:
            optimizer: The optimizer instance to update.
        """
        if self.best_q_params is not None:
            optimizer._q_params = self.best_q_params
        if self.best_c_params is not None:
            optimizer._c_params = self.best_c_params
