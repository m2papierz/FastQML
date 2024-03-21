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

from collections import OrderedDict
from typing import Dict
from typing import Any
from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt

import jax
from jax import vmap
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from fast_qml.core.estimator import Estimator


class FisherInformation:
    """
    Class encapsulating computation of the Fisher Information Matrix (FIM).

    For reference, see:
    Abbas et al., "The power of quantum neural networks."
    <https://arxiv.org/pdf/2011.00027.pdf>
    """

    argnums = {
        'quantum': 1, 'classical': 2, 'hybrid': (1, 2)
    }

    def __init__(
            self,
            estimator: Estimator,
            data: jnp.ndarray
    ):
        self._estimator = estimator
        self._fim = self._compute_fim(data)

    @property
    def fim(self):
        """
        Return the computed Fisher Information Matrix (FIM).
        """
        return self._fim

    @staticmethod
    def _ravel_quantum_grads(
            q_proba_d: OrderedDict,
            outputs_num: int
    ) -> Array:
        """
        Ravel quantum model parameters and concatenate into a single Array.

        Args:
            q_proba_d: Derivatives in regard to the quantum model parameters.
            outputs_num: Number of estimator outputs.

        Returns:
            Single Array with derivatives in regard to the quantum parameters.
        """
        if len(q_proba_d) > 1:
            q_proba_d = jnp.concatenate([
                grads for grads in q_proba_d.values()
            ], axis=1)
        else:
            q_proba_d = q_proba_d['QuantumModel0']
        q_proba_d = jnp.reshape(q_proba_d, newshape=(outputs_num, -1))

        return q_proba_d

    @staticmethod
    def _ravel_classical_grads(
            c_proba_d: Dict[str, Any],
            outputs_num: int
    ) -> Array:
        """
        Ravel classical model kernels and concatenate into a single Array.

        Args:
            c_proba_d: Derivatives in regard to the classical model parameters.
            outputs_num: Number of estimator outputs.

        Returns:
            Single Array with derivatives in regard to the classical parameters.
        """
        return jnp.concatenate(
            arrays=[
                jnp.reshape(layer['kernel'], newshape=(outputs_num, -1)).T
                for model_layer in c_proba_d.values()
                for layer in model_layer[0].values()
            ], axis=0
        )

    # @partial(jax.jit, static_argnums=0)
    def _get_proba_and_grads(
            self,
            x: ArrayLike
    ) -> Tuple[Array, Array]:
        """
        Computes the output probabilities and their gradients with respect to the estimator
         parameters for a given input.

        Args:
            x: The input data for which the Fisher Information Matrix is to be computed.

        Returns:
            A tuple containing two elements:
            - jnp.ndarray representing the output probabilities of the model for the given input.
            - jnp.ndarray representing the gradients of the output probabilities.
        """
        proba_d = jnp.array([])

        # Sample new set of model parameters and unpack them
        self._estimator.init_parameters(resample=True)
        q_params = self._estimator.q_parameters
        c_params = self._estimator.c_parameters

        # Compute model output probabilities
        proba = self._estimator.forward_pass(
            x_data=x,
            q_parameters=q_params,
            c_parameters=c_params,
            return_q_probs=True
        )

        # Compute derivatives of probabilities in regard to model parameters
        q_proba_d, c_proba_d = jax.jacfwd(
            self._estimator.forward_pass, argnums=(1, 2)
        )(x, q_params, c_params, True)

        # Ravel quantum parameters
        if len(q_proba_d) != 0:
            q_proba_d = self._ravel_quantum_grads(
                q_proba_d=q_proba_d, outputs_num=len(proba))
            proba_d = q_proba_d

        # Ravel classical parameters
        if len(c_proba_d) != 0:
            c_proba_d = self._ravel_classical_grads(
                c_proba_d=c_proba_d, outputs_num=len(proba))
            proba_d = c_proba_d

        # Concatenate quantum and classical parameters if both are applied
        if len(q_proba_d) != 0 and len(c_proba_d) != 0:
            proba_d = jnp.concatenate(arrays=[q_proba_d, c_proba_d.T], axis=1)

        return proba, proba_d

    @partial(jax.jit, static_argnums=0)
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

        return (proba_d.T * one_over_proba) @ proba_d

    @partial(jax.jit, static_argnums=0)
    def _compute_fim(
            self,
            data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the normalized Fisher Information Matrix (FIM) averaged over a batch of data.

        Args:
            data: The input data, a batch of observations for which the Fisher Information Matrix is
            to be computed.

        Returns:
            The normalized Fisher Information Matrix, averaged over the input batch of data.
        """
        # Create batched version of fisher matrix computation
        _compute_fisher_matrix_batched = vmap(
            self._compute_fisher_matrix, in_axes=0)

        # Compute FIM average over the given data
        fim = _compute_fisher_matrix_batched(x=data)

        # Normalize FIM
        fim = jnp.nanmean(fim, axis=0)
        fisher_inf_norm = fim * len(fim) / jnp.trace(fim)

        return fisher_inf_norm

    def plot_matrix(self) -> None:
        """
        Plots the Fisher Information Matrix (FIM) using a heatmap visualization.
        """
        sns.heatmap(self._fim, cmap="Greens", linewidths=0.8)
        plt.show()

    def plot_spectrum(self) -> None:
        """
        Plots a histogram of the eigenvalues of the Fisher Information Matrix (FIM). This
        method calculates the eigenvalues of the FIM and displays their distribution as a
        histogram, providing insight into the spectrum of the FIM.
        """
        eigenvalues = jnp.linalg.eigvalsh(self._fim)
        sns.histplot(eigenvalues, bins=len(eigenvalues), kde=False, alpha=0.85)
        plt.title("Histogram of the FIM eigenvalues")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Frequency")
        plt.show()
