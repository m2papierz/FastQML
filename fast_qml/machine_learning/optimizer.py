import jax
import jaxopt
import pennylane as qml

from jax import numpy as jnp
from functools import partial


class JITOptimizer:
    def __init__(
            self,
            params: jnp.ndarray,
            q_node: qml.qnode

    ):
        self._params = params
        self._q_node = q_node

    @partial(jax.jit, static_argnums=(0,))
    def _loss_fn(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
    ):
        predictions = self._q_node(x_data, weights)
        loss = jnp.sum((y_data - predictions) ** 2 / len(x_data))
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def optimize(
            self,
            data: jnp.ndarray,
            targets: jnp.ndarray,
            epochs: int
    ):
        def _compute_loss_and_grad(w, x, y, i):
            loss_val, grad_val = jax.value_and_grad(self._loss_fn)(w, x, y)
            jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
            return loss_val, grad_val

        opt = jaxopt.GradientDescent(_compute_loss_and_grad, stepsize=0.01, value_and_grad=True)
        opt_state = opt.init_state(self._params)

        def update(i, args):
            params, opt_state = opt.update(*args, i)
            return (params, opt_state, *args[2:])

        args = (self._params, opt_state, data, targets)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, epochs, update, args)

        return self._params
