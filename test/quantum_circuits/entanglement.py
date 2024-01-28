import pytest
import pennylane as qml
from fast_qml.quantum_circuits.entanglement import EntanglementGenerator


def apply_entanglement_pattern(entanglement_function):
    """Apply a given entanglement pattern function and return the resulting QuantumTape."""
    with qml.tape.QuantumTape() as tape:
        entanglement_function()
    return tape


# Entanglement pattern functions
def linear():
    """Generate a linear entanglement pattern."""
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[1, 2])


def reverse_linear():
    """Generate a reverse_linear entanglement pattern."""
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])


def circular():
    """Generate a circular entanglement pattern."""
    qml.CY(wires=[0, 1])
    qml.CY(wires=[1, 2])
    qml.CY(wires=[2, 0])


def reversed_circular():
    """Generate a reversed_circular entanglement pattern."""
    qml.CZ(wires=[2, 1])
    qml.CZ(wires=[1, 0])
    qml.CZ(wires=[0, 2])


def full():
    """Generate a full entanglement pattern."""
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])


# User-defined entanglement patterns
def custom_entanglement_4_qubits():
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[1, 2])
    qml.CZ(wires=[0, 2])
    qml.CZ(wires=[1, 3])


def custom_entanglement_5_qubits():
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[4, 1])


@pytest.mark.parametrize("entanglement_pattern", [
    ("linear", linear, 'CZ'),
    ("reverse_linear", reverse_linear, 'CX'),
    ("circular", circular, 'CY'),
    ("reverse_circular", reversed_circular, 'CZ'),
    ("full", full, 'CX')
])
def test_entanglement_patterns(entanglement_pattern):
    pattern_name, expected_func, c_gate = entanglement_pattern
    entangler = EntanglementGenerator(
        n_qubits=3, c_gate=c_gate, entanglement=pattern_name)

    tape_expected = apply_entanglement_pattern(expected_func)
    tape_test = apply_entanglement_pattern(entangler.apply)

    assert set(tape_expected) == set(tape_test)


@pytest.mark.parametrize("entanglement_pattern", [
    (4, 'CZ', [[0, 1], [1, 2], [0, 2], [1, 3]], custom_entanglement_4_qubits),
    (5, 'CX', [[0, 1], [4, 1]], custom_entanglement_5_qubits)
])
def test_user_defined_pattern(entanglement_pattern):
    qubits, c_gate, pattern, expected_func = entanglement_pattern
    entangler = EntanglementGenerator(
        n_qubits=qubits, c_gate=c_gate, entanglement=pattern)

    tape_expected = apply_entanglement_pattern(expected_func)
    tape_test = apply_entanglement_pattern(entangler.apply)

    assert set(tape_expected) == set(tape_test)
