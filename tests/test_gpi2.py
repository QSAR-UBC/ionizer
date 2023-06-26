"""
Unit tests for GPI2 operation and its utility functions.
"""
import pytest
import torch
import jax

import pennylane as qml
import numpy as np
import tensorflow as tf
from ionizer.ops import GPI2

jax.config.update("jax_enable_x64", True)

class TestGPI2:
    """Tests for the Operations ``GPI2``."""

    interfaces = [
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
        pytest.param("tf", marks=pytest.mark.tf)
    ]

    @staticmethod
    def circuit(x):
        GPI2(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    @staticmethod
    def interface_array(x, interface):
        """Create a trainable array of x in the specified interace."""
        if interface is None:
            return x
        if interface == "autograd":
            return qml.numpy.array(x)
        if interface == "jax":
            jax.config.update("jax_enable_x64", True)
            return jax.numpy.array(x)
        if interface == "torch":
            return torch.tensor(x)
        if interface == "tf":
            return tf.Variable(x)
        return None
    
    @pytest.mark.parametrize("phi, expected", [(0, 1/np.sqrt(2) * (np.array([[1, -1j],[-1j, 1]]))), (0.3, 1/np.sqrt(2)*np.array([[1, -1j * np.exp(-1j * 0.3)],[-1j * np.exp(1j * 0.3), 1]]))])
    def test_compute_matrix_returns_correct_matrix(self, phi, expected):
       assert  qml.math.allequal(GPI2.compute_matrix(phi), expected)

    @pytest.mark.parametrize("interface", interfaces)
    def test_compute_gradient(self, interface):
        phi = 0.3
        phi = self.interface_array(phi, interface)
        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(self.circuit, dev, interface=interface)
        phi_grad = None
        if interface == "jax":
            grad_fcn = jax.grad(qnode)
            assert grad_fcn(phi) == 4.3661002990646334e-17
        elif interface == "torch":
            phi.requires_grad = True
            result = qnode(phi)
            result.backward()
            assert float(phi.grad) == 2.9802322387695312e-08
        elif interface == "tf":
            with tf.GradientTape() as tape:
                loss = qnode(phi)
            assert float(tape.gradient(loss, [phi])[0]) == 0.0
        else:
            assert qml.grad(qnode, argnum=0)(phi) == 5.551115123125783e-17