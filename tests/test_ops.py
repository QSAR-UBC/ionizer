"""
Unit tests for GPI2 operation and its utility functions.
"""

import pytest
import torch
import jax

import pennylane as qml
from pennylane import numpy as np
import pennylane.math as math
import tensorflow as tf
from ionizer.ops import GPI

jax.config.update("jax_enable_x64", True)

interfaces_and_array_methods = [
    ["autograd", qml.numpy.array],
    ["torch", torch.tensor],
    ["tf", tf.Variable],
    ["jax", jax.numpy.array],
]
diff_methods = ["backprop", "adjoint", "parameter-shift", "finite-diff"]
two_pi = 2 * np.pi


def get_GPI_matrix(phi):
    """TODO"""
    return np.array([[0, math.exp(-1j * phi)], [math.exp(1j * phi), 0]])


class State:
    @staticmethod
    def set_state():
        """TODO"""
        qml.RX(0.2, wires=0)
        qml.RY(1.1, wires=0)
        qml.RX(0.3, wires=0)

    @classmethod
    @qml.qnode(qml.device("default.qubit", wires=1))
    def get_state(cls):
        """TODO"""
        cls.set_state()
        return qml.state()

    def __init__(self):
        """TODO"""
        self.state = self.get_state()
        self.a_conj_b = self.state[0] * np.conj(self.state[1])
        self.b_conj_a = self.state[1] * np.conj(self.state[0])


@pytest.mark.parametrize("phi", [0, 0.37 * two_pi, 1.23 * two_pi, two_pi])
class TestGPI:
    @staticmethod
    def circuit(phi):
        """TODO"""
        State.set_state()
        GPI(phi, wires=0)
        return qml.expval(qml.PauliY(wires=0))

    @pytest.fixture(autouse=True)
    def state(self):
        """TODO"""
        self._state = State()

    def test_GPI_compute_matrix(self, phi):
        """TODO"""
        gpi_matrix = GPI.compute_matrix(phi)
        check_matrix = get_GPI_matrix(phi_value)

        assert math.allclose(gpi_matrix, check_matrix)

    @pytest.mark.parametrize("interface, array_method", interfaces_and_array_methods)
    def test_GPI_circuit(self, interface, array_method, phi):
        """TODO"""
        phi_GPI = array_method(phi)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface)
        val_GPI = qnode_GPI(phi_GPI)

        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi)
        expected_val = expected_inner_product_1 + expected_inner_product_2

        assert np.isclose(
            val_GPI, expected_val, atol=1e-07
        ), f"Given val: {val_GPI}; Expected val: {expected_val}"

    def get_circuit_grad(self, phi):
        """TODO"""
        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi) * (-2j)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi) * 2j
        return np.real(expected_inner_product_1 + expected_inner_product_2)

    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_GPI_grad_qnode(self, diff_method, phi):
        """TODO"""

        phi_GPI = np.array(phi)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, diff_method=diff_method)
        grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"

    @pytest.mark.autograd
    def test_GPI_grad_autograd(self, phi):
        """TODO"""
        phi_GPI = qml.numpy.array(phi, requires_grad=True)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface, diff_method=diff_method)
        grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"

    @pytest.mark.jax
    def test_GPI_grad_jax(self, phi):
        """TODO"""
        import jax

        phi_GPI = jax.numpy.array(phi, dtype=jax.numpy.complex128)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface, diff_method=diff_method)
        grad_GPI = jax.grad(qnode_GPI)(phi_GPI)

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"

    @pytest.mark.torch
    def test_GPI_grad_torch(self, phi):
        """TODO"""
        import torch

        phi_GPI = torch.tensor(phi, requires_grad=True)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface, diff_method=diff_method)

        phi_GPI.requires_grad = True
        result = qnode_GPI(phi_GPI)
        result.backward()
        grad_GPI = phi_GPI.grad

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"

    @pytest.mark.tf
    def test_GPI_grad_tensorflow(self, phi):
        """TODO"""
        import tensorflow as tf

        phi_GPI = tf.Variable(phi)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface, diff_method=diff_method)

        with tf.GradientTape() as tape:
            loss = qnode_GPI(phi_GPI)
        grad_GPI = tape.gradient(loss, phi_GPI)

        expected_grad = self.get_circuit_grad(phi)
        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"