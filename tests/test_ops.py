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

interfaces = ["autograd", "torch", "tf", "jax"]
diff_methods = ["backprop", "adjoint", "parameter-shift", "finite-diff"]
two_pi = 2 * np.pi


def get_GPI_matrix(phi):
    return np.array([[0, math.exp(-1j * phi)], [math.exp(1j * phi), 0]])


class State:
    @staticmethod
    def set_state():
        qml.RX(0.2, wires=0)
        qml.RY(1.1, wires=0)
        qml.RX(0.3, wires=0)

    @classmethod
    @qml.qnode(qml.device("default.qubit", wires=1))
    def get_state(cls):
        cls.set_state()
        return qml.state()

    def __init__(self):
        self.state = self.get_state()
        self.a_conj_b = self.state[0] * np.conj(self.state[1])
        self.b_conj_a = self.state[1] * np.conj(self.state[0])


class TestGPI:
    @staticmethod
    def circuit(phi):
        State.set_state()
        GPI(phi, wires=0)
        return qml.expval(qml.PauliY(wires=0))

    @staticmethod
    def interface_array(x, interface):
        match interface:
            case "jax":
                jax.config.update("jax_enable_x64", True)
                return jax.numpy.array(x)
            case "torch":
                return torch.tensor(x)
            case "tf":
                return tf.Variable(x)
            case "autograd":
                return qml.numpy.array(x)

    @pytest.fixture(autouse=True)
    def state(self):
        self._state = State()

    @pytest.mark.parametrize("interface", interfaces)
    def test_GPI_compute_matrix(self, interface):
        phi_rand = np.random.rand() * two_pi
        phi_values = [0, phi_rand, two_pi]

        for phi_value in phi_values:
            phi_interface = self.interface_array(phi_value, interface)
            gpi_matrix = GPI.compute_matrix(phi_interface)

            check_matrix = get_GPI_matrix(phi_value)

            assert math.allclose(gpi_matrix, check_matrix)

    @pytest.mark.parametrize("interface", interfaces)
    def test_GPI_circuit(self, interface, phi):
        phi_GPI = self.interface_array(phi, interface)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface)
        val_GPI = qnode_GPI(phi_GPI)

        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi)
        expected_val = expected_inner_product_1 + expected_inner_product_2

        assert np.isclose(
            val_GPI, expected_val, atol=1e-07
        ), f"Given val: {val_GPI}; Expected val: {expected_val}"

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("phi", [0.37 * two_pi, 1.23 * two_pi, two_pi])
    def test_GPI_grad(self, diff_method, interface):
        phi = np.random.rand() * two_pi
        phi_GPI = self.interface_array(phi, interface)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(
            self.circuit, dev, interface=interface, diff_method=diff_method
        )

        match interface:
            case "torch":
                phi_GPI.requires_grad = True
                result = qnode_GPI(phi_GPI)
                result.backward()
                grad_GPI = phi_GPI.grad

            case "tf":
                with tf.GradientTape() as tape:
                    loss = qnode_GPI(phi_GPI)
                grad_GPI = tape.gradient(loss, phi_GPI)
            case "jax":
                grad_GPI = jax.grad(qnode_GPI)(phi_GPI)

            case "autograd":
                grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)

        expected_inner_product_1 = 1j * self._state.b_conj_a * np.exp(-2j * phi) * (-2j)
        expected_inner_product_2 = -1j * self._state.a_conj_b * np.exp(2j * phi) * 2j
        expected_grad = np.real(expected_inner_product_1 + expected_inner_product_2)

        assert np.isclose(
            grad_GPI, expected_grad
        ), f"Given grad: {grad_GPI}; Expected grad: {expected_grad}"
