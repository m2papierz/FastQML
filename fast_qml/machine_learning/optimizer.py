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
import jaxopt
import pennylane as qml

from jax import tree_util
from jax import numpy as jnp
from abc import abstractmethod


class Optimizer:
    def __init__(
            self,
            params: jnp.ndarray,
            q_node: qml.qnode

    ):
        self._params = params
        self._q_node = q_node

    @abstractmethod
    def _loss_fn(self, weights, x_data, y_data):
        pass

    @abstractmethod
    def _compute_loss_and_grad(self, params, data, target, epoch):
        pass

    @abstractmethod
    def optimize(self, data, targets, epochs):
        pass


class DefaultOptimizer(Optimizer):
    def __init__(
            self,
            params: jnp.ndarray,
            q_node: qml.qnode

    ):
        super().__init__(params=params, q_node=q_node)

    def _loss_fn(self, weights, x_data, y_data):
        pass

    def _compute_loss_and_grad(self, params, data, target, epoch):
        pass

    def optimize(self, data, targets, epochs):
        pass


class JITOptimizer(Optimizer):
    def __init__(
            self,
            params: jnp.ndarray,
            q_node: qml.qnode

    ):
        super().__init__(params=params, q_node=q_node)

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
            weights: jnp.ndarray,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
    ):
        predictions = self._q_node(x_data, weights)
        loss = jnp.sum((y_data - predictions) ** 2 / len(x_data))
        return loss

    @jax.jit
    def _compute_loss_and_grad(
            self,
            params: jnp.ndarray,
            data: jnp.ndarray,
            target: jnp.ndarray,
            epoch: int
    ):
        loss_val, grad_val = jax.value_and_grad(
            self._loss_fn)(params, data, target)
        jax.debug.print(
            "Epoch: {i} - Loss: {loss}", i=epoch, loss=loss_val)
        return loss_val, grad_val

    @jax.jit
    def optimize(
            self,
            data: jnp.ndarray,
            targets: jnp.ndarray,
            epochs: int
    ):
        opt = jaxopt.GradientDescent(
            self._compute_loss_and_grad, stepsize=0.01, value_and_grad=True)
        opt_state = opt.init_state(self._params)

        def update(i, args):
            params, opt_state = opt.update(*args, i + 1)
            return (params, opt_state, *args[2:])

        args = (self._params, opt_state, data, targets)
        (params, opt_state, _, _) = jax.lax.fori_loop(
            lower=0, upper=epochs,
            body_fun=update, init_val=args
        )

        return self._params
