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
An implementation of the Fisher Information Matrix (FIM).
"""

from functools import partial
from dataclasses import asdict

import jax
from jax import vmap
import jax.numpy as jnp

from fast_qml.core.estimator import Estimator


class FisherInformation:
    """
    Class encapsulating computation of the Fisher Information Matrix (FIM).

    For reference, see:
    Abbas et al., "The power of quantum neural networks."
    <https://arxiv.org/pdf/2011.00027.pdf>
    """
    def __init__(
            self,
            estimator: Estimator
    ):
        self._estimator = estimator

    def _get_proba_and_grads(
            self,
            x: jnp.ndarray
    ):
        """
        Computes the output probabilities and their gradients with respect to the model
        parameters for a given input.

        Args:
            x: The input data for which the Fisher Information Matrix is to be computed.

        Returns:
            A tuple containing two elements:
            - jnp.ndarray representing the output probabilities of the model for the given input.
            - jnp.ndarray representing the gradients of the output probabilities.
        """
        # Sample new set of model parameters and unpack them
        self._estimator.init_parameters()
        c_params, q_params, batch_stats, _ = asdict(self._estimator.params).values()

        # Ensure that input dimension is correct of the classical estimator component
        if self._estimator.input_shape is not None:
            if len(x.shape) < len(self._estimator.input_shape):
                x = jnp.expand_dims(x, axis=0)

        # Compute model output probabilities
        proba = self._estimator.model(
            x_data=x, q_weights=q_params,
            c_weights=c_params, batch_stats=batch_stats,
            training=False, q_model_probs=True)

        # Depending on estimator type compute jacobian over different parameters
        estimator_type = self._estimator.estimator_type
        if estimator_type == "quantum":
            argnums = 1
        elif estimator_type == "classical":
            argnums = 2
        else:
            argnums = (1, 2)

        # Compute derivatives of probabilities in regard to model parameters
        proba_d = jax.jacfwd(
            self._estimator.model,
            argnums=argnums
        )(x, q_params, c_params, batch_stats, False, True)

        return proba, proba_d

    def _compute_fisher_matrix(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the Fisher Information Matrix for a given input.

        Args:
            x: The input data for which the Fisher Information Matrix is to be computed.

        Returns:
            The Fisher Information Matrix, a square array of shape (n_params, n_params), where
            `n_params` is the number of model outputs (observables).
        """
        # Compute probabilities and gradients
        proba, proba_d = self._get_proba_and_grads(x=x)

        # Exclude zero values and calculate 1 / proba
        non_zeros_proba = jnp.where(
            proba > 0, proba, jnp.ones_like(proba))
        one_over_proba = jnp.where(
            proba > 0, jnp.ones_like(proba), jnp.zeros_like(proba))
        one_over_proba = one_over_proba / non_zeros_proba

        # Cast, reshape, and transpose matrix to get (n_params, n_params) array
        proba_d = jnp.asarray(proba_d, dtype=proba.dtype)
        proba_d = jnp.reshape(proba_d, newshape=(len(proba), -1))
        proba_d_over_p = jnp.transpose(proba_d) * one_over_proba

        return proba_d_over_p @ proba_d

    @partial(jax.jit, static_argnums=0)
    def fisher_information(
            self,
            x_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the normalized Fisher Information Matrix (FIM) averaged over a batch of data.

        Args:
            x_data: The input data, a batch of observations for which the Fisher Information Matrix is
            to be computed.

        Returns:
            The normalized Fisher Information Matrix, averaged over the input batch of data.
        """
        # Create batched version of fisher matrix computation
        _compute_fisher_matrix_batched = vmap(
            self._compute_fisher_matrix, in_axes=0)

        # Compute FIM average over the given data
        fim = jnp.nanmean(_compute_fisher_matrix_batched(x_data), axis=0)

        # Normalize FIM
        fisher_inf_norm = fim * len(fim) / jnp.trace(fim)

        return fisher_inf_norm
