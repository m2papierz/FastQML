# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

"""
An implementation of the estimator Effective Dimension.
"""

from typing import Union
import jax.numpy as jnp

from scipy.special import logsumexp
from fast_qml.core.estimator import Estimator
from fast_qml.estimator_analysis.fisher_information import FisherInformation


class EffectiveDimension:
    """
    Class encapsulating computation of the Effective Dimension for an Estimator.

    For reference, see:
    Abbas et al., "The power of quantum neural networks."
    <https://arxiv.org/pdf/2011.00027.pdf>
    """
    def __init__(
            self,
            estimator: Estimator,
            fisher_information_matrix: Union[jnp.ndarray, None] = None
    ):
        self._estimator = estimator
        self._fim = fisher_information_matrix

    def _set_fisher_matrix(
            self,
            x_data: jnp.ndarray
    ) -> None:
        """
        Computes the fisher information matrix if it was not provided.

        Args:
            x_data: The input data for which the Fisher Information Matrix is to be computed.

        Returns:
            The normalized Fisher Information Matrix, averaged over the input data.
        """
        if self._fim is None:
            self._fim = FisherInformation(
                estimator=self._estimator, data=x_data).fim

    def get_effective_dimension(
            self,
            x_data: jnp.ndarray
    ) -> float:
        """
        Computes the effective dimension based on the Fisher Information Matrix for the
        given estimator and data.

        Args:
            x_data: The input data, a batch of observations for which the Fisher Information Matrix is
            to be computed, and based on which Effective Dimension is to be computed.

        Returns:
            Effective dimension for a given estimator and dataset.
        """
        dataset_size = jnp.array(x_data.shape[0])
        self._set_fisher_matrix(x_data)

        # Matrix of which determinant will be calculated incorporating FIM
        fim_mod = self._fim * dataset_size / (2 * jnp.pi * jnp.log(dataset_size))
        one_plus_fmod = jnp.eye(len(self._fim)) + fim_mod

        # Take logarithm of the determinant
        det_log = jnp.linalg.slogdet(one_plus_fmod)[1] / 2

        # Compute effective dimension
        numerator = logsumexp(det_log, axis=None) - jnp.log(len(self._fim))
        denominator = jnp.log(dataset_size / (2 * jnp.pi * jnp.log(dataset_size)))
        effective_dims = jnp.squeeze(2 * numerator / denominator)
        effective_dims = effective_dims / self._estimator.params_num

        return float(effective_dims)
