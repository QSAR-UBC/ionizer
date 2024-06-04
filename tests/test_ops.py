"""
Unit tests for GPI2 operation and its utility functions.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
import pennylane.math as math
from ionizer.ops import GPI

diff_methods = ["backprop", "adjoint", "parameter-shift", "finite-diff"]
two_pi = 2 * np.pi


def get_GPI_matrix(phi):
    """Gets the GPI expected GPI matrix"""
    return np.array([[0, math.exp(-1j * phi)], [math.exp(1j * phi), 0]])


class State:
    @staticmethod
    def set_state():
        """Quantum function to set a known starting state to use for testing."""
        qml.RX(0.2, wires=0)
        qml.RY(1.1, wires=0)
        qml.RX(0.3, wires=0)

    @classmethod
    @qml.qnode(qml.device("default.qubit", wires=1))
    def get_state(cls):
        """Gets the predefined starting state"""
        cls.set_state()
        return qml.state()

    def __init__(self):
        """Creates an object for an intial quantum state."""
        self.state = self.get_state()
        self.a_conj_b = self.state[0] * np.conj(self.state[1])
        self.b_conj_a = self.state[1] * np.conj(self.state[0])


@pytest.mark.parametrize("phi", [0, 0.37 * two_pi, 1.23 * two_pi, two_pi])
class TestGPI:
    @staticmethod
    def circuit(phi):
        """Circuit that applies the GPI gate to a known starting state
        returning the expectation value of Pauli Y."""
        State.set_state()
        GPI(phi, wires=0)
        return qml.expval(qml.PauliY(wires=0))

    @pytest.fixture(autouse=True)
    def state(self):
        """Starting state to be used in computation"""
        self._state = State()

    def test_GPI_compute_matrix(self, phi):
        """Tests wheter the right matrix is returned depending on phi."""
        gpi_matrix = GPI.compute_matrix(phi)
        check_matrix = get_GPI_matrix(phi)

        assert math.allclose(gpi_matrix, check_matrix)

    def test_GPI_circuit(self, array_method, phi):
        """Tests whether the expected value is returned for a circuit using GPI"""
        phi_GPI = array_method(phi)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev)
        val_GPI = qnode_GPI(phi_GPI)

        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi)
        expected_val = expected_inner_product_1 + expected_inner_product_2

        assert np.isclose(
            val_GPI, expected_val, atol=1e-07
        ), f"Given val: {val_GPI}; Expected val: {expected_val}"

    def get_circuit_grad(self, phi):
        """Gets the expected gradient of the basic circuit using GPI's output."""
        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi) * (-2j)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi) * 2j
        return np.real(expected_inner_product_1 + expected_inner_product_2)

    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_GPI_grad(self, diff_method, phi):
        """Tests whether the correct gradient for a circuit using GPI
        using different gradient computation methods."""

        phi_GPI = np.array(phi)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, diff_method=diff_method)
        grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"
