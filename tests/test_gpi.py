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

interfaces = [
        "autograd",
        "torch",
        "tf",
        "jax",
        "auto",
        None
    ]
two_pi = 2 * np.pi

def get_GPI_matrix(phi):
    return np.array([[0, math.exp(-1j * phi)], [math.exp(1j * phi), 0]])


class TestGPI:
    @staticmethod
    def circuit(phi): 
        qml.Hadamard(wires=0)
        GPI(phi, wires=0)
        return qml.expval(qml.PauliZ(wires=0))
    
    @staticmethod
    def circuit_unitary(phi): 
        qml.Hadamard(wires=0)
        gpi_matrix = get_GPI_matrix(phi)
        qml.QubitUnitary(gpi_matrix, wires=0)
        return qml.expval(qml.PauliZ(wires=0))
    
    @staticmethod
    def interface_array(x, interface):
        match interface:
            case None:
                return x
            case "jax":
                jax.config.update("jax_enable_x64", True)
                return jax.numpy.array(x)
            case "torch":
                return torch.tensor(x)
            case "tf":
                return tf.Variable(x)
            case _:
                return qml.numpy.array(x)
            
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
    def test_GPI_circuit(self, interface): 
        phi = np.random.rand() * two_pi 
        phi_GPI = self.interface_array(phi, interface)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface)
        qnode_unitary = qml.QNode(self.circuit_unitary, dev)

        val_unitary = qnode_unitary(phi)
        val_GPI = qnode_GPI(phi_GPI)

        assert np.isclose(val_GPI, val_unitary)


    @pytest.mark.parametrize("interface", interfaces)
    def test_GPI_grad(self, interface):
        phi = np.random.rand() * two_pi
        phi_GPI = self.interface_array(phi, interface)
        dev = qml.device("default.qubit", wires=1)

        qnode_GPI = qml.QNode(self.circuit, dev, interface=interface)
        qnode_unitary = qml.QNode(self.circuit_unitary, dev)

        grad_unitary = qml.grad(qnode_unitary, argnum=0)(phi)

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

            case None:
                with pytest.raises(Exception):
                    grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)
                return
            
            case _:
                grad_GPI = qml.grad(qnode_GPI, argnum=0)(phi_GPI)
        assert np.isclose(grad_GPI, grad_unitary)

