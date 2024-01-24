import jax
import jaxopt
import pennylane as qml

from jax import numpy as jnp
from functools import partial

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm


class VariationalModel:
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            n_layers: int = 1
    ):
        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._ansatz._reps = n_layers

        self._device = qml.device("default.qubit.jax", wires=n_qubits)

        self._params = self._init_weights()

    def _init_weights(self) -> jnp.ndarray:
        params_num = int(self._ansatz.get_params_num())
        return jax.random.normal(jax.random.PRNGKey(42), shape=(params_num,))

    def _q_model(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray
    ):
        @jax.jit
        @qml.qnode(device=self._device, interface='jax')
        def _quantum_circuit():
            self._feature_map.apply(features=x_data)
            self._ansatz.apply(params=weights)
            return qml.expval(qml.PauliZ(0))
        return _quantum_circuit()

    @partial(jax.jit, static_argnums=(0,))
    def _loss_fn(
            self,
            weights: jnp.ndarray,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray
    ):
        predictions = self._q_model(x_data, weights)
        loss = jnp.sum((y_data - predictions) ** 2 / len(x_data))
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def fit(
            self,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
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

        args = (self._params, opt_state, x_data, y_data)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, epochs, update, args)
