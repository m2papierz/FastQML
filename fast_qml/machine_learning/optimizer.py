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
from pennylane import numpy as qnp


class Optimizer:
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            learning_rate: float

    ):
        self._q_node = q_node
        self._params = params
        self._learning_rate = learning_rate

    @abstractmethod
    def _loss_fn(self, weights, x_data, y_data):
        pass

    @abstractmethod
    def optimize(self, data, targets, verbose, epochs):
        pass


class DefaultOptimizer(Optimizer):
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            learning_rate: float

    ):
        super().__init__(
            params=params,
            q_node=q_node,
            learning_rate=learning_rate
        )

    def _loss_fn(self, weights, x_data, y_data):
        predictions = self._q_node(weights=weights, x_data=x_data)
        loss = qnp.sum((y_data - predictions) ** 2 / len(x_data))
        return loss

    def optimize(self, data, targets, verbose, epochs):
        _opt = qml.AdamOptimizer(self._learning_rate)
        for it in range(epochs):
            self._params = _opt.step(self._loss_fn, self._params, x_data=data, y_data=targets)
            loss_val = self._loss_fn(self._params, data, targets)

            print(f"Epoch: {it} - Loss: {loss_val}")

        return self._params


class JITOptimizer(Optimizer):
    def __init__(
            self,
            params: np.ndarray,
            q_node: qml.qnode,
            learning_rate: float
    ):
        super().__init__(
            params=params,
            q_node=q_node,
            learning_rate=learning_rate
        )
        self._opt = optax.adam(learning_rate=learning_rate)

    def _tree_flatten(self):
        children = (self._params,)
        aux_data = {
            'q_node': self._q_node,
            'learning_rate': self._learning_rate
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(
            *children,
            q_node=aux_data['q_node'],
            learning_rate=aux_data['learning_rate']
        )

    @classmethod
    def register_pytree_node(cls):
        tree_util.register_pytree_node(
            nodetype=cls,
            flatten_func=cls._tree_flatten,
            unflatten_func=cls._tree_unflatten
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
            jax.debug.print("Epoch: {i} - Loss: {loss_val}", i=i, loss_val=loss_val)
        jax.lax.cond(print_training, print_fn, lambda: None)

        return params, opt_state, data, targets, print_training

    @jax.jit
    def optimize(
            self,
            data: np.ndarray,
            targets: np.ndarray,
            verbose: bool,
            epochs: int
    ):
        opt_state = self._opt.init(self._params)
        args = (self._params, opt_state, data, targets, verbose)
        (self._params, opt_state, _, _, _) = jax.lax.fori_loop(
            lower=0, upper=epochs, body_fun=self._update_step, init_val=args)
        return self._params
