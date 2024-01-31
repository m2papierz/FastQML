# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.


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
