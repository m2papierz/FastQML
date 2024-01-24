# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import jax
import optax
import numpy as np
import pennylane as qml

from jax import tree_util
from abc import abstractmethod

from jax import numpy as jnp


class Optimizer:
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode

    ):
        self._q_node = q_node
        self._params = params

    @abstractmethod
    def _loss_fn(self, weights, x_data, y_data):
        pass

    @abstractmethod
    def optimize(self, data, targets, epochs):
        pass


class JITOptimizer(Optimizer):
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode

    ):
        super().__init__(params=params, q_node=q_node)
        self._opt = optax.adam(learning_rate=0.3)

    def _tree_flatten(self):
        children = (self._params,)
        aux_data = {'q_node': self._q_node}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, q_node=aux_data['q_node'])

    @classmethod
    def register_pytree_node(cls):
        tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten
        )

    @jax.jit
    def _loss_fn(
            self,
            weights: np.ndarray,
            x_data: np.ndarray,
            y_data: np.ndarray
    ):
        predictions = self._q_node(weights=weights, x_data=x_data)
        loss = jnp.sum((y_data - predictions) ** 2 / len(x_data))
        return loss

    @jax.jit
    def _update_step(self, i, args):
        params, opt_state, data, targets, print_training = args

        loss_val, grads = jax.value_and_grad(self._loss_fn)(params, data, targets)
        updates, opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        def print_fn():
            jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

        # if print_training=True, print the loss every 5 steps
        jax.lax.cond(print_training, print_fn, lambda: None)

        return params, opt_state, data, targets, print_training

    @jax.jit
    def optimize(
            self,
            data: np.ndarray,
            targets: np.ndarray,
            learning_rate: float,
            verbose: bool,
            epochs: int
    ):
        opt = optax.adam(learning_rate=learning_rate)
        opt_state = opt.init(self._params)

        args = (self._params, opt_state, data, targets, verbose)
        (params, opt_state, _, _, _) = jax.lax.fori_loop(
            0, epochs, self._update_step, args)

        return self._params
