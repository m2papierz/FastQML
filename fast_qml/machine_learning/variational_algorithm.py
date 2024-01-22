import jax
import jaxopt
import pennylane as qml

from typing import Callable

from jax import numpy as jnp

from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm


class VariationalQuantumAlgorithm:
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_operator: qml.Operation,
            loss_fn: Callable,
            device: str = 'CPU'
    ):
        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._measurement_op = measurement_operator
        self._loss_fn = loss_fn

        self._device = self._config_device(device)
        self._params = {'weights': ansatz._params}

    def _config_device(self, device: str):
        if device == 'CPU':
            jax.config.update("jax_platform_name", "cpu")
            return qml.device("default.qubit", wires=self._n_qubits)
        elif device == 'GPU':
            return qml.device("lightning.qubit", wires=self._n_qubits)
        else:
            raise ValueError()

    def _circuit(self):
        @qml.qnode(device=self._device)
        def _quantum_circuit():
            self._feature_map.circuit()
            self._ansatz.circuit()
            return qml.expval(op=self._measurement_op)
        return _quantum_circuit()

    @jax.jit
    def _optimization_loop_cpu(
            self,
            x_data: jnp.ndarray,
            y_data: jnp.ndarray,
            epochs: int,
            verbose: bool = False
    ):
        def loss_and_grad(idx):
            loss_val, grad_val = jax.value_and_grad(
                self._loss_fn)(self._params, x_data, y_data)
            jax.debug.print(f"Step: {idx} - Loss: {loss_val}")
            return loss_val, grad_val

        opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.1, value_and_grad=True)
        opt_state = opt.init_state(self._params)

        for i in range(epochs):
            self._params, opt_state = opt.update(
                self._params, opt_state, x_data, y_data, True, i)
